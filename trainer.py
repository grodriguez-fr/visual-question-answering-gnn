#!/usr/bin/env python
#-*- coding:utf-8 -*-

import json, os, sys, time
import torch
import utils.utils as utils
# Import conditionnel du logger
try:
    from utils.logger import Logger
    LOGGER_AVAILABLE = True
except Exception as e:
    LOGGER_AVAILABLE = False
    print(f"⚠️ Logger non disponible (TensorBoard désactivé): {e}")
    Logger = None
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
from IPython.display import display, Image
from tqdm import tqdm

import logging, time, platform
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb non disponible, désactivé automatiquement")

class Trainer():
    """Trainer object that defines the training loop and evaluation logic

    opt: Parser object representing the parameters entered by user
    train_loader: Object containing the batched training dataset
    val_loader: Object containing the batched validation dataset
    model: Object representing the Seq2Seq model for this task
    executor: Object used to execute the CLEVR queries using question and image.
    """

    def __init__(self, opt, train_loader, val_loader, model, executor):
        self.opt = opt
        self.reinforce = opt.reinforce
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        self.visualize_training_wandb = opt.visualize_training_wandb
        if opt.dataset == 'clevr':
            self.vocab = utils.load_vocab(opt.clevr_vocab_path)
        elif opt.dataset == 'clevr-humans':
            self.vocab = utils.load_vocab(opt.human_vocab_path)
        else:
            raise ValueError('Invalid dataset')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.executor = executor

        # Create Optimizer #
        # _params_bline = list(filter(lambda p: p.requires_grad, model.seq2seq_baseline.parameters()))
        _params = list(filter(lambda p: p.requires_grad, model.seq2seq.parameters()))
        _params_gnn = list(filter(lambda p: p.requires_grad, model.seq2seq.gnn.parameters()))
        _params_enc = list(filter(lambda p: p.requires_grad, model.seq2seq.encoder.parameters()))
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.seq2seq.parameters()),
        #                                   lr=opt.learning_rate)
        self.optimizer = torch.optim.Adam(_params, lr=opt.learning_rate)
        self.stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0
        }
        if opt.visualize_training:
            if not LOGGER_AVAILABLE:
                print("⚠️ Logger/TensorBoard requis mais non disponible")
                opt.visualize_training = 0
            else:
                # Tensorboard - créer un nouveau dossier avec timestamp pour chaque entraînement #
                from utils.logger import Logger
                import time
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                # Construire un nom descriptif avec toutes les infos importantes
                gnn_type = opt.gnn.upper()  # GAT ou GIN
                stream_mode = "two-stream" if opt.two_stream_gnn else "single-stream"
                log_name = f'logs_seq2seq_{gnn_type}_{stream_mode}_{timestamp}'
                log_dir = f'{opt.run_dir}/{log_name}'
                self.logger = Logger(log_dir)
                print(f"📊 TensorBoard logs: {log_dir}")

        if opt.visualize_training_wandb:
            if not WANDB_AVAILABLE:
                print("⚠️ wandb requis mais non disponible, TensorBoard sera utilisé à la place")
                opt.visualize_training_wandb = 0
            else:
                # WandB: Log metrics with wandb #
                wandb_proj_name = opt.wandb_proj_name
                wandb_identifier = opt.run_identifier
                wandb_name = f"{wandb_identifier}"
                wandb.init(project=wandb_proj_name, name=wandb_name, notes="Running from mgn.reason.trainer.py")
                wandb.config.update(opt)
                wandb.watch(self.model.seq2seq)

    def train(self):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        num_workers = self.opt.num_workers
        t = 0
        epoch = 0
        baseline = 0
        start = time.time()
        
        # Calcul du nombre d'epochs et d'itérations par epoch
        batch_size = self.opt.batch_size
        train_size = len(self.train_loader.dataset)
        iters_per_epoch = train_size // batch_size
        total_epochs = (self.num_iters + iters_per_epoch - 1) // iters_per_epoch
        
        while t < self.num_iters:
            epoch += 1
            # Barre de progression pour cet epoch
            epoch_desc = f"Epoch {epoch}/{total_epochs}"
            pbar = tqdm(self.train_loader, desc=epoch_desc, total=min(iters_per_epoch, self.num_iters - t))
            
            # Retrieve current batch from the train loader object
            for batch in pbar:
                t += 1
                loss, reward = None, None
                # Extract data from batch (handle both dict and tuple formats)
                if isinstance(batch, dict):
                    x = batch['question']
                    y = batch.get('program', None)
                    ans = batch.get('answer', None)
                    idx = batch.get('image_idx', None)
                    g_data = None  # Graph data not in dict format
                else:
                    x, y, ans, idx, g_data = batch
                # Send input to the model
                self.model.set_input(x, y, g_data)
                self.optimizer.zero_grad()
                if self.reinforce:
                    pred = self.model.reinforce_forward()
                    reward = self.get_batch_reward(pred, ans, idx, 'train')
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    self.model.reinforce_backward(self.entropy_factor)
                else:
                    loss = self.model.supervised_forward()
                    self.model.supervised_backward()
                # Take a step in the direction that reduces the gradient fastest
                self.optimizer.step()

                if t % self.display_every == 0:
                    if self.reinforce:
                        self.stats['train_batch_accs'].append(reward)
                        self.log_stats('training batch reward', reward, t)
                        pbar.set_postfix({'reward': f'{reward:.4f}'})
                    else:
                        self.stats['train_losses'].append(loss)
                        self.log_stats('training batch loss', loss, t)
                        
                        # Calculer training batch accuracy
                        train_batch_acc = self.get_batch_reward(self.model.parse(), ans, idx, 'train', x)
                        self.stats['train_batch_accs'].append(train_batch_acc)
                        self.log_stats('training batch accuracy', train_batch_acc, t)
                        
                        pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{train_batch_acc:.4f}'})
                    self.stats['train_accs_ts'].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    # Run model on the validation set
                    val_acc = self.check_val_accuracy()
                    pbar.write(f'| checkpoint at iter {t}/{self.num_iters}, val_acc: {val_acc:.2%}')
                    
                    # Log validation accuracy to TensorBoard
                    self.log_stats('validation accuracy', val_acc, t)
                    
                    if val_acc >= self.stats['best_val_acc']:
                        pbar.write('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        checkpoint_fp = f"{self.run_dir}/checkpoint_best.pt"
                        self.model.save_checkpoint(checkpoint_fp)
                        self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' % (self.run_dir, t))
                        if self.visualize_training_wandb:
                            wandb.save(checkpoint_fp)
                    
                    # Toujours calculer et logger le validation loss en mode supervisé
                    if not self.reinforce:
                        val_loss = self.check_val_loss()
                        pbar.write('| validation loss %f' % val_loss)
                        self.stats['val_losses'].append(val_loss)
                        self.log_stats('validation loss', val_loss, t)
                    self.stats['val_accs'].append(val_acc)
                    self.stats['val_accs_ts'].append(t)
                    # Save Checkpoint #
                    self.model.save_checkpoint('%s/checkpoint.pt' % self.run_dir)

                    checkpoint_dict = {
                        'args': self.opt.__dict__,
                        'stats': self.stats
                    }
                    json_fp = '%s/stats.json' % self.run_dir
                    #json_fp = os.path.join(self.run_dir, f'stats_{self.opt.run_timestamp}.json')
                    logging.info(f'Saving train_opt_[ts].json at: {json_fp}')
                    with open(json_fp, 'w') as f:
                        json.dump(self.stats, f, indent=2)
                        #json.dump(checkpoint_dict, f, indent=1)
                    if self.visualize_training_wandb:
                        wandb.save(json_fp)
                    self.log_params(t)

                if t >= self.num_iters:
                    pbar.close()
                    break
            
            # Fermer la barre de progression de l'epoch si pas déjà fermée
            if t < self.num_iters:
                pbar.close()
                
        end = time.time()
        # logging.info(f"Finished {self.num_iters} or {epoch}epochs in {end-start}s, with {num_workers}workers")  # Désactivé

    def check_val_loss(self):
        loss = 0
        t = 0
        for batch in self.val_loader:
            # Extract data from batch (handle both dict and tuple formats)
            if isinstance(batch, dict):
                x = batch['question']
                y = batch.get('program', None)
                g_data = None
            else:
                x, y, _, _, g_data = batch
            self.model.set_input(x, y, g_data)
            loss += self.model.supervised_forward()
            t += 1
        return loss / t if t != 0 else 0

    def check_val_accuracy(self):
        output_file = open("curr_valid_predictions.txt", "w")
        reward = 0
        t = 0
        for batch in self.val_loader:
            # Extract data from batch (handle both dict and tuple formats)
            if isinstance(batch, dict):
                x = batch['question']
                y = batch.get('program', None)
                ans = batch.get('answer', None)
                idx = batch.get('image_idx', None)
                g_data = None
            else:
                x, y, ans, idx, g_data = batch
            self.model.set_input(x, y, g_data)
            pred = self.model.parse()
            reward += self.get_batch_reward(pred, ans, idx, 'val', x)
            t += 1
        reward = reward / t if t != 0 else 0
        return reward

    def get_batch_reward(self, programs, answers, image_idxs, split, input):
        pg_np = programs.numpy()
        ans_np = answers.numpy()
        idx_np = image_idxs.numpy()
        reward = 0
        # Protection contre les batches vides
        if pg_np.shape[0] == 0:
            return 0.0
        for i in range(pg_np.shape[0]):
            pred = self.executor.run(pg_np[i], idx_np[i], split)
            ans = self.vocab['answer_idx_to_token'][ans_np[i]]
            if pred == ans:
                reward += 1.0
            if pred != 'error':
                input_str = ""
                for index in range(min(pg_np[i].shape[0], len(input[i]))):  # Limiter à la longueur de input[i]
                    if input[i][index] > 0:
                        input_str += self.vocab['question_idx_to_token'][int(input[i][index])] + " "
                # print(input_str)  # Debug output disabled
                img_num = ("000000" + str(int(image_idxs[i])))[-6:]
                # print("Prediction = " + pred)  # Debug output disabled
                # print("Answer = " + ans)  # Debug output disabled
                output_file = open("curr_valid_predictions.txt", "a")
                output_file.write(pred + " " + ans + " " + img_num + " " + input_str + "\n")
        reward /= pg_np.shape[0]
        return reward

    def log_stats(self, tag, value, t):
        if self.visualize_training_wandb and WANDB_AVAILABLE:
            wandb.log({tag: value}, step=t)

        # Temsorboard #
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)


    def log_params(self, t):
        if self.visualize_training_wandb and WANDB_AVAILABLE:
            train_qfn = self.opt.clevr_train_question_filename
            if train_qfn and not self.reinforce:
                wandb.log({"Info": wandb.Table(data=[train_qfn], columns=["train_question_filename"])})
            for k,v in self.stats.items():
                wandb.log({k: v}, step=t)

        # Tensorboard #
        if self.visualize_training and self.logger is not None:
            for tag, value in self.model.seq2seq.named_parameters():
                tag = tag.replace('.', '/')
                # Protection contre les tenseurs vides
                value_np = self._to_numpy(value)
                if value_np.size > 0:
                    self.logger.histo_summary(tag, value_np, t)
                if value.grad is not None:
                    grad_np = self._to_numpy(value.grad)
                    if grad_np.size > 0:
                        self.logger.histo_summary('%s/grad' % tag, grad_np, t)


    def _to_numpy(self, x):
        return x.data.cpu().numpy()