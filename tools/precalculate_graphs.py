#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Précalcule tous les graphes pour les questions CLEVR et les sauvegarde
pour accélérer l'entraînement
"""

import os
import sys
import json
import pickle
import h5py
from tqdm import tqdm
import argparse

# Ajouter le workspace au path
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace)

def precalculate_graphs(questions_json_path, scenes_json_path, output_pickle_path, 
                       max_samples=None):
    """
    Précalcule les graphes pour toutes les questions et les sauvegarde
    
    Args:
        questions_json_path: Chemin vers le JSON des questions
        scenes_json_path: Chemin vers le JSON des scènes
        output_pickle_path: Chemin de sortie pour le pickle des graphes
        max_samples: Nombre max d'échantillons (None = tous)
    """
    from datasets.clevr_questions import graph_parser, embedder
    
    print(f"📊 Précalcul des graphes")
    print(f"   - Questions: {questions_json_path}")
    print(f"   - Scènes: {scenes_json_path}")
    print(f"   - Output: {output_pickle_path}")
    print()
    
    # Charger les questions
    with open(questions_json_path, 'r') as f:
        questions_data = json.load(f)
    questions = questions_data['questions']
    
    if max_samples is not None:
        questions = questions[:max_samples]
    
    print(f"📝 {len(questions)} questions à traiter")
    
    # Charger les scènes
    with open(scenes_json_path, 'r') as f:
        scenes_data = json.load(f)
    scenes = {scene['image_index']: scene for scene in scenes_data['scenes']}
    
    print(f"🖼️ {len(scenes)} scènes chargées")
    print()
    
    # Précalculer les graphes
    graphs_cache = {}
    
    print("🔄 Génération des graphes...")
    for q in tqdm(questions, desc="Graphes"):
        img_idx = q['image_index']
        question_text = q['question']
        
        # Clé unique pour chaque question
        cache_key = f"{img_idx}_{question_text}"
        
        if cache_key in graphs_cache:
            continue
        
        # Récupérer la scène
        scene = scenes.get(img_idx)
        if scene is None:
            print(f"⚠️ Scène {img_idx} introuvable, skip")
            continue
        
        try:
            # Générer le graphe
            graph_data = graph_parser.get_graph_data(
                scene=scene,
                question=question_text,
                embedder=embedder
            )
            
            graphs_cache[cache_key] = graph_data
            
        except Exception as e:
            print(f"❌ Erreur pour question {img_idx}: {e}")
            graphs_cache[cache_key] = None
    
    print()
    print(f"✅ {len(graphs_cache)} graphes générés")
    
    # Sauvegarder
    print(f"💾 Sauvegarde dans {output_pickle_path}...")
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(graphs_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Afficher la taille du fichier
    size_mb = os.path.getsize(output_pickle_path) / (1024 * 1024)
    print(f"✅ Graphes sauvegardés ({size_mb:.1f} MB)")
    print()
    
    return graphs_cache


def main():
    parser = argparse.ArgumentParser(description='Précalcule les graphes CLEVR')
    parser.add_argument('--questions_json', type=str, required=True,
                       help='Chemin vers le JSON des questions')
    parser.add_argument('--scenes_json', type=str, required=True,
                       help='Chemin vers le JSON des scènes')
    parser.add_argument('--output_pickle', type=str, required=True,
                       help='Chemin de sortie pour le pickle')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Nombre max d\'échantillons (optionnel)')
    
    args = parser.parse_args()
    
    precalculate_graphs(
        questions_json_path=args.questions_json,
        scenes_json_path=args.scenes_json,
        output_pickle_path=args.output_pickle,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
