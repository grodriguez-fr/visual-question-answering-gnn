#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test rapide du trainer pour débugger sans notebook
"""

import os
import sys
import time

# Fix Windows encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configuration paths
workspace = r'd:\visual-question-answering-gnn'
os.chdir(workspace)
sys.path.insert(0, workspace)

# Forcer CPU
os.environ['FORCE_CPU_MODE'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("🚀 Test rapide du trainer")
print("=" * 80)

# ====== ÉTAPE 0: Vérifier les graphes précalculés ======
print("\n🔍 Vérification des graphes précalculés...")
train_graph_file = 'clevr_preprocessed_10k/clevr_train_questions_directed_pairdata.pt'
val_graph_file = 'clevr_preprocessed_10k/clevr_val_questions_directed_pairdata.pt'

if os.path.exists(train_graph_file) and os.path.exists(val_graph_file):
    train_size = os.path.getsize(train_graph_file) / (1024 * 1024)
    val_size = os.path.getsize(val_graph_file) / (1024 * 1024)
    print(f"✅ Graphes précalculés trouvés:")
    print(f"   - Train: {train_size:.1f} MB")
    print(f"   - Val: {val_size:.1f} MB")
else:
    print("⚠️  Graphes précalculés non trouvés")
    print("   Les graphes seront générés dynamiquement (plus lent)")
    print("   Pour précalculer: exécutez `python precalculate_graphs_10k.py`")

print()

# Import des modules
from options.train_options import TrainOptions
from datasets import ClevrQuestionDataset, ClevrQuestionDataLoader
from datasets.clevr_questions import graph_parser, embedder
from models.parser import Seq2seqParser
from executors import get_executor
from trainer import Trainer

# Configuration
sys.argv = ['']
opt = TrainOptions().parse()

# IMPORTANT: Corriger TOUS les chemins après le parse()
opt.clevr_vocab_path = r'd:\visual-question-answering-gnn\clevr_preprocessed_10k\clevr_vocab.json'
opt.clevr_train_question_path = r'd:\visual-question-answering-gnn\clevr_preprocessed_10k\clevr_train_questions.h5'
opt.clevr_val_question_path = r'd:\visual-question-answering-gnn\clevr_preprocessed_10k\clevr_val_questions.h5'
opt.clevr_train_scene_path = r'd:\visual-question-answering-gnn\data\CLEVR_v1.0\scenes\CLEVR_train_scenes.json'
opt.clevr_val_scene_path = r'd:\visual-question-answering-gnn\data\CLEVR_v1.0\scenes\CLEVR_val_scenes.json'
opt.clevr_train_parsed_scene_path = r'd:\visual-question-answering-gnn\data\CLEVR_v1.0\scenes\CLEVR_train_scenes.json'
opt.clevr_val_parsed_scene_path = r'd:\visual-question-answering-gnn\data\CLEVR_v1.0\scenes\CLEVR_val_scenes.json'
opt.clevr_train_raw_question_path = r'd:\visual-question-answering-gnn\clevr_preprocessed_10k\temp\train_questions_8k.json'
opt.clevr_val_raw_question_path = r'd:\visual-question-answering-gnn\clevr_preprocessed_10k\temp\val_questions_2k.json'

# Configuration entraînement
opt.gembd_vec_dim = 96  # ACTIVER LE GNN avec 96 dimensions
opt.max_train_samples = 8000
opt.max_val_samples = 2000
opt.batch_size = 32
opt.num_iters = 50  # Test rapide avec seulement 50 iterations (~2 epochs)
opt.checkpoint_every = 250
opt.display_every = 10  # Afficher plus souvent
opt.visualize_training = 0  # Désactiver TensorBoard pour éviter conflit tensorflow
opt.visualize_training_wandb = 0
opt.num_workers = 0
opt.is_train = True

print(f"✅ Configuration chargée:")
print(f"   - Train samples: {opt.max_train_samples}")
print(f"   - Val samples: {opt.max_val_samples}")
print(f"   - Batch size: {opt.batch_size}")
print(f"   - Itérations: {opt.num_iters} (~2 epochs)")
print(f"   🔥 GNN ACTIVÉ: {opt.gembd_vec_dim}D embeddings ({opt.gnn})")
print()

# Chargement des données
print("📊 Chargement des données...")
train_dataset = ClevrQuestionDataset(opt, 'train', graph_parser=graph_parser, embedder=embedder)
val_dataset = ClevrQuestionDataset(opt, 'val', graph_parser=graph_parser, embedder=embedder)

train_loader = ClevrQuestionDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
val_loader = ClevrQuestionDataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

print(f"✅ Données chargées: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
print()

# Création du modèle
print("🏗️ Construction du modèle...")
model = Seq2seqParser(opt)
executor = get_executor(opt)
print(f"✅ Modèle créé")
print()

# Test d'un batch
print("🧪 Test d'un batch...")
try:
    for batch_idx, (x, prog, y, img_idx, g_data) in enumerate(train_loader):
        print(f"✅ Batch {batch_idx + 1} chargé avec succès!")
        print(f"   - Questions shape: {x.shape}")
        print(f"   - Programs shape: {prog.shape}")
        print(f"   - Answers shape: {y.shape}")
        print(f"   - Image indices shape: {img_idx.shape}")
        if g_data and g_data[0] is not None:
            print(f"   - Graph data: {g_data}")
        break
except Exception as e:
    print(f"❌ Erreur lors du chargement du batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Initialisation du trainer
print("🚀 Initialisation du Trainer...")
trainer = Trainer(opt, train_loader, val_loader, model, executor)
print("✅ Trainer initialisé")
print()

# Lancement de l'entraînement
print("🏃 Lancement de l'entraînement...")
print("=" * 80)
start_time = time.time()

try:
    trainer.train()
    print(f"\n✅ Entraînement terminé avec succès!")
    if trainer.stats['train_losses']:
        print(f"   - Loss initiale: {trainer.stats['train_losses'][0]:.6f}")
        print(f"   - Loss finale: {trainer.stats['train_losses'][-1]:.6f}")
        print(f"   - Best val accuracy: {trainer.stats['best_val_acc']:.2f}%")
except KeyboardInterrupt:
    print(f"\n⚠️ Entraînement interrompu par l'utilisateur")
except Exception as e:
    print(f"\n❌ Erreur durant l'entraînement: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

elapsed = time.time() - start_time
print(f"\n⏱️ Temps total: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
print("=" * 80)
