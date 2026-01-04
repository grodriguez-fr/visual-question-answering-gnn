#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lance le précalcul des graphes pour le dataset 10k (8000 train + 2000 val)
Utilise tqdm pour afficher la progression
"""

import subprocess
import sys
import os

# Fix Windows encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("🚀 Précalcul des graphes pour 10k échantillons")
print("=" * 80)

# Chemins vers le venv Python
python_exe = sys.executable  # Utiliser l'interpréteur Python actuel

# Vérifier si les graphes existent déjà
# Les noms doivent correspondre exactement aux noms des fichiers h5
train_graph_file = 'clevr_preprocessed_10k/clevr_train_questions_directed_pairdata.pt'
val_graph_file = 'clevr_preprocessed_10k/clevr_val_questions_directed_pairdata.pt'

train_exists = os.path.exists(train_graph_file)
val_exists = os.path.exists(val_graph_file)

if train_exists and val_exists:
    train_size = os.path.getsize(train_graph_file) / (1024 * 1024)
    val_size = os.path.getsize(val_graph_file) / (1024 * 1024)
    print("\n✅ Les graphes sont déjà précalculés!")
    print(f"   - Train: {train_graph_file} ({train_size:.1f} MB)")
    print(f"   - Val: {val_graph_file} ({val_size:.1f} MB)")
    print("\n⏩ Précalcul ignoré (fichiers déjà présents)")
    print("=" * 80)
    sys.exit(0)

# ====== TRAIN: 8000 échantillons ======
if not train_exists:
    print("\n📊 Précalcul des graphes TRAIN (8000 échantillons)...")
    print("-" * 80)

    train_cmd = [
        python_exe,
        "tools/sample_questions_mgn.py",
        "--random_sample", "1",
        "--max_sample", "8000",
        "--input_question_h5", "clevr_preprocessed_10k/clevr_train_questions.h5",
        "--output_dir", "clevr_preprocessed_10k",
        "--input_questions_json", "clevr_preprocessed_10k/temp/train_questions_8k.json",
        "--input_parsed_img_scenes_json", "data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json",
        "--input_vocab_json", "clevr_preprocessed_10k/clevr_vocab.json",
        "--is_directed_graph", "1"
    ]

    print(f"Commande: {' '.join(train_cmd)}\n")

    result = subprocess.run(train_cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n❌ Erreur lors du précalcul des graphes train")
        sys.exit(1)

    print("\n✅ Graphes train précalculés!")
else:
    print(f"\n✅ {train_graph_file} existe déjà, skip")

# ====== VAL: 2000 échantillons ======
if not val_exists:
    print("\n📊 Précalcul des graphes VAL (2000 échantillons)...")
    print("-" * 80)

    val_cmd = [
        python_exe,
        "tools/sample_questions_mgn.py",
        "--random_sample", "1",
        "--max_sample", "2000",
        "--input_question_h5", "clevr_preprocessed_10k/clevr_val_questions.h5",
        "--output_dir", "clevr_preprocessed_10k",
        "--input_questions_json", "clevr_preprocessed_10k/temp/val_questions_2k.json",
        "--input_parsed_img_scenes_json", "data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json",
        "--input_vocab_json", "clevr_preprocessed_10k/clevr_vocab.json",
        "--is_directed_graph", "1"
    ]

    print(f"Commande: {' '.join(val_cmd)}\n")

    result = subprocess.run(val_cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n❌ Erreur lors du précalcul des graphes val")
        sys.exit(1)

    print("\n✅ Graphes val précalculés!")
else:
    print(f"\n✅ {val_graph_file} existe déjà, skip")

# ====== Vérification ======
print("\n📋 Vérification des fichiers générés:")
print("-" * 80)

expected_files = [
    "clevr_preprocessed_10k/clevr_train_8000_questions_directed_pairdata.pt",
    "clevr_preprocessed_10k/clevr_val_2000_questions_directed_pairdata.pt",
]

for f in expected_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"✅ {f} ({size_mb:.1f} MB)")
    else:
        print(f"❌ MANQUANT: {f}")

print("\n" + "=" * 80)
print("🎉 Précalcul terminé!")
print("\nLes graphes sont sauvegardés et seront chargés automatiquement")
print("par le dataset si les fichiers *_pairdata.pt existent.")
