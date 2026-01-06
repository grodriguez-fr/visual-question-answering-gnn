#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Trainer for Classification Mode (End-to-End VQA)

This trainer is used when classification_mode=1, bypassing the executor
and predicting answers directly from graph embeddings.
"""

import json, os, sys, time
import torch
import utils.utils as utils
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

# Import conditionnel du logger
try:
    from utils.logger import Logger
    LOGGER_AVAILABLE = True
except Exception as e:
    LOGGER_AVAILABLE = False
    Logger = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ClassifierTrainer:
    """
    Trainer for end-to-end VQA classification.
    
    Unlike the standard Trainer, this one:
    - Uses CrossEntropyLoss on answer labels (not program tokens)
    - Computes accuracy directly without executor
    - Does not generate programs
    
    Args:
        opt: Options object
        train_loader: Training data loader
        val_loader: Validation data loader
        model: ClassifierModel instance
    """

    def __init__(self, opt, train_loader, val_loader, model):
        self.opt = opt
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        self.visualize_training_wandb = getattr(opt, 'visualize_training_wandb', 0)
        
        # Load vocabulary
        if opt.dataset == 'clevr':
            self.vocab = utils.load_vocab(opt.clevr_vocab_path)
        else:
            raise ValueError('Invalid dataset')
        
        # Build answer mappings
        self.answer_idx_to_token = {v: k for k, v in self.vocab['answer_token_to_idx'].items()}
        self.num_classes = len(self.vocab['answer_token_to_idx'])
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        
        # Create Optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.learning_rate
        )
        
        self.stats = {
            'train_losses': [],
            'train_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0
        }
        
        # TensorBoard setup
        if opt.visualize_training:
            if not LOGGER_AVAILABLE:
                print("⚠️ Logger/TensorBoard requis mais non disponible")
                opt.visualize_training = 0
            else:
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                # Construire un nom descriptif avec toutes les infos importantes
                gnn_type = opt.gnn.upper()  # GAT ou GIN
                stream_mode = "two-stream" if opt.two_stream_gnn else "single-stream"
                log_name = f'logs_classifier_{gnn_type}_{stream_mode}_{timestamp}'
                log_dir = f'{opt.run_dir}/{log_name}'
                self.logger = Logger(log_dir)
                print(f"📊 TensorBoard logs: {log_dir}")
        
        # WandB setup
        if self.visualize_training_wandb and WANDB_AVAILABLE:
            wandb_proj_name = getattr(opt, 'wandb_proj_name', 'vqa_classifier')
            wandb_name = f"classifier_{time.strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=wandb_proj_name, name=wandb_name)
            wandb.config.update(opt)

    def train(self):
        """Main training loop"""
        print('| 🎯 Starting CLASSIFICATION training in', self.run_dir)
        print(f'| Mode: Direct answer prediction ({self.num_classes} classes)')
        
        t = 0
        epoch = 0
        start = time.time()
        
        # Calculate epochs
        batch_size = self.opt.batch_size
        train_size = len(self.train_loader.dataset)
        iters_per_epoch = max(1, train_size // batch_size)
        total_epochs = (self.num_iters + iters_per_epoch - 1) // iters_per_epoch
        
        self.model.train()
        
        while t < self.num_iters:
            epoch += 1
            epoch_desc = f"Epoch {epoch}/{total_epochs}"
            pbar = tqdm(self.train_loader, desc=epoch_desc, 
                       total=min(iters_per_epoch, self.num_iters - t))
            
            for batch in pbar:
                t += 1
                
                # Extract data from batch
                if isinstance(batch, dict):
                    x = batch.get('question', None)
                    ans = batch.get('answer', None)
                    g_data = None
                else:
                    x, y, ans, idx, g_data = batch
                
                # Set input (we need g_data and answers)
                self.model.set_input(x=x, g_data=g_data, answers=ans)
                
                # Forward + backward
                self.optimizer.zero_grad()
                loss = self.model.supervised_forward()
                self.model.supervised_backward()
                self.optimizer.step()
                
                # Compute batch accuracy
                with torch.no_grad():
                    predictions = self.model.predict()
                    batch_acc = (predictions == ans).float().mean().item()
                
                # Logging
                if t % self.display_every == 0:
                    self.stats['train_losses'].append(loss)
                    self.stats['train_accs'].append(batch_acc)
                    self.stats['train_accs_ts'].append(t)
                    self.log_stats('training batch loss', loss, t)
                    self.log_stats('training batch accuracy', batch_acc, t)
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{batch_acc:.2%}'})
                
                # Validation checkpoint
                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    self.model.eval()
                    val_acc, val_loss = self.check_val_accuracy()
                    self.model.train()
                    
                    pbar.write(f'| checkpoint at iter {t}/{self.num_iters}, val_acc: {val_acc:.2%}, val_loss: {val_loss:.4f}')
                    
                    self.log_stats('validation accuracy', val_acc, t)
                    self.log_stats('validation loss', val_loss, t)
                    
                    self.stats['val_accs'].append(val_acc)
                    self.stats['val_losses'].append(val_loss)
                    self.stats['val_accs_ts'].append(t)
                    
                    # Save best model
                    if val_acc >= self.stats['best_val_acc']:
                        pbar.write('| 🏆 best model!')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        self.model.save_checkpoint(f"{self.run_dir}/checkpoint_best.pt")
                        self.model.save_checkpoint(f'{self.run_dir}/checkpoint_iter{t:08d}.pt')
                    
                    # Save current checkpoint
                    self.model.save_checkpoint(f'{self.run_dir}/checkpoint.pt')
                    
                    # Save stats
                    with open(f'{self.run_dir}/stats.json', 'w') as f:
                        json.dump(self.stats, f, indent=2)
                
                if t >= self.num_iters:
                    pbar.close()
                    break
            
            if t < self.num_iters:
                pbar.close()
        
        end = time.time()
        print(f'\n| ✅ Training completed in {end-start:.1f}s')
        print(f'| Best validation accuracy: {self.stats["best_val_acc"]:.2%}')

    def check_val_accuracy(self):
        """
        Evaluate on validation set.
        
        Returns:
            accuracy: Float accuracy on validation set
            avg_loss: Average loss on validation set
        """
        total_correct = 0
        total_samples = 0
        total_loss = 0
        num_batches = 0
        
        # Open predictions file
        output_file = open("curr_valid_predictions.txt", "w")
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extract data
                if isinstance(batch, dict):
                    x = batch.get('question', None)
                    ans = batch.get('answer', None)
                    idx = batch.get('image_idx', None)
                    g_data = None
                else:
                    x, y, ans, idx, g_data = batch
                
                if ans is None:
                    continue
                
                # Forward pass
                self.model.set_input(x=x, g_data=g_data, answers=ans)
                loss = self.model.supervised_forward()
                predictions = self.model.predict()
                
                # Compute accuracy
                correct = (predictions == ans).sum().item()
                total_correct += correct
                total_samples += ans.size(0)
                total_loss += loss
                num_batches += 1
                
                # Write predictions to file
                for i in range(len(predictions)):
                    pred_ans = self.answer_idx_to_token.get(predictions[i].item(), '<UNK>')
                    true_ans = self.answer_idx_to_token.get(ans[i].item(), '<UNK>')
                    
                    # Build question string if available
                    q_str = ""
                    if x is not None:
                        for token_idx in x[i]:
                            if token_idx > 0:
                                token = self.vocab['question_idx_to_token'].get(int(token_idx), '')
                                q_str += token + " "
                    
                    img_num = str(int(idx[i])).zfill(6) if idx is not None else "000000"
                    output_file.write(f"{pred_ans} {true_ans} {img_num} {q_str.strip()}\n")
        
        output_file.close()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return accuracy, avg_loss

    def log_stats(self, tag, value, t):
        """Log statistics to TensorBoard and/or WandB"""
        if self.visualize_training_wandb and WANDB_AVAILABLE:
            wandb.log({tag: value}, step=t)
        
        if self.visualize_training and hasattr(self, 'logger') and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)
