"""
Visualisation XAI pour le modèle MGN.
Affiche le graphe de question, l'image de la scène et le graphe de scène avec attention GAT.
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from PIL import Image
from collections import defaultdict
import logging

# Désactiver TOUS les logs de debug PIL et Matplotlib
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PngImagePlugin').setLevel(logging.ERROR)


def visualize_gat_attention(gat_model, x, edge_index, layer_idx=0, head_idx=0, 
                            node_labels=None, edge_relations=None, vocab=None,
                            title="GAT Attention", 
                            figsize=(12, 10), threshold=0.05):
    """Visualise le graphe d'attention d'une couche GAT."""
    gat_model.eval()
    with torch.no_grad():
        out, attn_list = gat_model(x, edge_index, return_attention=True)
        edge_idx_layer, alpha = attn_list[layer_idx]
        attn_weights = alpha[:, head_idx].cpu().numpy()
        
    sources = edge_idx_layer[0].cpu().numpy()
    targets = edge_idx_layer[1].cpu().numpy()
    
    G = nx.DiGraph()
    num_nodes = x.shape[0]
    
    for i in range(num_nodes):
        label = node_labels[i] if node_labels and i < len(node_labels) else f"N{i}"
        G.add_node(i, label=label)
    
    edges_to_draw = []
    colors = []
    widths = []
    edge_labels_dict = {}
    
    for idx, (s, t, w) in enumerate(zip(sources, targets, attn_weights)):
        if w > threshold:
            G.add_edge(s, t, weight=w)
            edges_to_draw.append((s, t))
            colors.append(w)
            widths.append(1 + w * 3)
            
            edge_label = f"{w:.2f}"
            if edge_relations and (s, t) in edge_relations:
                rel_name = edge_relations[(s, t)]
                if isinstance(rel_name, str) and not rel_name.startswith('rel'):
                    edge_label = f"{rel_name}\n{w:.2f}"
            
            edge_labels_dict[(s, t)] = edge_label
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if num_nodes <= 15:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.spring_layout(G, k=1.8, iterations=100, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8, ax=ax)
    
    if edges_to_draw:
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=widths, 
                              edge_color=colors, edge_cmap=plt.cm.Reds, 
                              edge_vmin=0, edge_vmax=1,
                              arrowstyle='->', arrowsize=15,
                              connectionstyle='arc3,rad=0.1', ax=ax)
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict,
                                     font_size=7, font_color='darkred', ax=ax)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Intensité de l'attention", shrink=0.8)
    
    ax.set_title(f"{title} (Couche {layer_idx}, Tête {head_idx})", fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def calculate_spatial_relation(obj1_pos, obj2_pos):
    """Calcule la relation spatiale entre deux objets."""
    dx = obj2_pos[0] - obj1_pos[0]
    dy = obj2_pos[1] - obj1_pos[1]
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    else:
        return "behind" if dy < 0 else "front"


def visualize_graph_structure(x, edge_index, node_labels=None, edge_relations=None, 
                              title="Graph Structure", figsize=(12, 10)):
    """Visualise la structure du graphe sans attention (pour GIN, GCN, etc.)"""
    sources = edge_index[0].cpu().numpy()
    targets = edge_index[1].cpu().numpy()
    
    G = nx.DiGraph()
    num_nodes = x.shape[0]
    
    for i in range(num_nodes):
        label = node_labels[i] if node_labels and i < len(node_labels) else f"N{i}"
        G.add_node(i, label=label)
    
    edge_labels_dict = {}
    for idx, (s, t) in enumerate(zip(sources, targets)):
        G.add_edge(s, t)
        if edge_relations and (s, t) in edge_relations:
            rel_name = edge_relations[(s, t)]
            if isinstance(rel_name, str) and not rel_name.startswith('rel'):
                edge_labels_dict[(s, t)] = rel_name
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if num_nodes <= 15:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.spring_layout(G, k=1.8, iterations=100, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray', 
                          arrowstyle='->', arrowsize=15,
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    if edge_labels_dict:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict,
                                     font_size=7, font_color='darkred', ax=ax)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_sample(model, val_loader, val_dataset, opt, sample_idx=0):
    """
    Visualise un échantillon: graphe de question, image de scène, graphe de scène et programme généré.
    
    Args:
        model: Le modèle MGN
        val_loader: DataLoader de validation
        val_dataset: Dataset de validation
        opt: Options du modèle
        sample_idx: Index de l'échantillon dans le batch (0 par défaut)
    """
    # Chargement des données
    batch = next(iter(val_loader))
    questions, programs, answers, image_idxs, g_data_batch = batch
    BATCH_S, BATCH_T = g_data_batch
    
    idx_to_extract = sample_idx
    vocab = val_dataset.vocab
    
    # Extraction des graphes
    mask_s = BATCH_S.batch == idx_to_extract
    mask_t = BATCH_T.batch == idx_to_extract
    
    x_question = BATCH_S.x[mask_s]
    x_scene = BATCH_T.x[mask_t]
    
    edge_mask_s = mask_s[BATCH_S.edge_index[0]] & mask_s[BATCH_S.edge_index[1]]
    edge_index_s = BATCH_S.edge_index[:, edge_mask_s]
    
    edge_mask_t = mask_t[BATCH_T.edge_index[0]] & mask_t[BATCH_T.edge_index[1]]
    edge_index_t = BATCH_T.edge_index[:, edge_mask_t]
    
    # Renumérotation des nœuds
    node_mapping_s = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.where(mask_s)[0])}
    node_mapping_t = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.where(mask_t)[0])}
    
    edge_index_s_remapped = torch.tensor([
        [node_mapping_s[src.item()] for src in edge_index_s[0]],
        [node_mapping_s[tgt.item()] for tgt in edge_index_s[1]]
    ], dtype=torch.long)
    
    edge_index_t_remapped = torch.tensor([
        [node_mapping_t[src.item()] for src in edge_index_t[0]],
        [node_mapping_t[tgt.item()] for tgt in edge_index_t[1]]
    ], dtype=torch.long)
    
    # Métadonnées
    question_tokens = questions[idx_to_extract].cpu().numpy()
    question_words_all = []
    for token_idx in question_tokens:
        if token_idx in vocab['question_idx_to_token']:
            word = vocab['question_idx_to_token'][token_idx]
            if word == '<END>':
                break
            if word not in ['<NULL>', '<START>']:
                question_words_all.append(word)
    
    image_idx = image_idxs[idx_to_extract].item()
    
    OUTPUT_DIR = f'clevr_preprocessed_{opt.max_train_samples + opt.max_val_samples}'
    val_questions_path = os.path.join(OUTPUT_DIR, 'temp', f'val_questions_{opt.max_val_samples}.json')
    with open(val_questions_path, 'r') as f:
        val_questions_json = json.load(f)['questions']
    
    question_data = val_questions_json[idx_to_extract]
    image_filename = question_data['image_filename']
    
    with open(opt.clevr_val_scene_path, 'r') as f:
        scenes_data = json.load(f)
    
    scene = None
    for s in scenes_data['scenes']:
        if s['image_index'] == image_idx:
            scene = s
            break
    
    if scene:
        scene_objects = []
        scene_objects_full = []
        for obj in scene['objects']:
            size = obj.get('size', '')
            color = obj.get('color', '')
            shape = obj.get('shape', '')
            material = obj.get('material', '')
            
            label = f"{color[:3]}{shape[:3]}"
            scene_objects.append(label)
            
            full_label = f"{size} {material} {color} {shape}"
            scene_objects_full.append(full_label)
    else:
        scene_objects = [f"O{i}" for i in range(x_scene.shape[0])]
        scene_objects_full = scene_objects
    
    # Création des labels de question
    question_labels = []
    if len(question_words_all) <= x_question.shape[0]:
        for i in range(x_question.shape[0]):
            if i < len(question_words_all):
                question_labels.append(question_words_all[i])
            else:
                question_labels.append(f"syntrel_{i}")
    else:
        words_per_node = len(question_words_all) / x_question.shape[0]
        for i in range(x_question.shape[0]):
            start_idx = int(i * words_per_node)
            end_idx = int((i + 1) * words_per_node)
            node_words = question_words_all[start_idx:end_idx]
            if node_words:
                question_labels.append(' '.join(node_words))
            else:
                question_labels.append(f"Q{i}")
    
    # Analyse de la structure de scène
    node_out_connections = defaultdict(set)
    node_in_connections = defaultdict(set)
    edge_relations = {}
    
    temp_out_connections = defaultdict(set)
    for idx, (src, tgt) in enumerate(edge_index_t_remapped.t().tolist()):
        temp_out_connections[src].add(tgt)
    
    object_nodes = [i for i in range(x_scene.shape[0]) if len(temp_out_connections[i]) > 0]
    
    obj_positions = {}
    if scene and len(scene['objects']) > 0:
        for obj_idx, obj in enumerate(scene['objects']):
            if obj_idx < len(object_nodes):
                node_idx = object_nodes[obj_idx]
                obj_positions[node_idx] = obj['3d_coords']
    
    for idx, (src, tgt) in enumerate(edge_index_t_remapped.t().tolist()):
        node_out_connections[src].add(tgt)
        node_in_connections[tgt].add(src)
        
        if src in object_nodes and tgt in object_nodes:
            if src in obj_positions and tgt in obj_positions:
                spatial_rel = calculate_spatial_relation(obj_positions[src], obj_positions[tgt])
                edge_relations[(src, tgt)] = spatial_rel
            else:
                edge_relations[(src, tgt)] = "spatial"
        elif src in object_nodes and tgt not in object_nodes:
            obj_attrs = [t for s, t in edge_index_t_remapped.t().tolist() 
                        if s == src and t not in object_nodes]
            attr_position = obj_attrs.index(tgt) if tgt in obj_attrs else 0
            
            attr_types = ['hasSize', 'hasColor', 'hasMaterial', 'hasShape']
            if attr_position < len(attr_types):
                edge_relations[(src, tgt)] = attr_types[attr_position]
            else:
                edge_relations[(src, tgt)] = "hasAttr"
        else:
            edge_relations[(src, tgt)] = "other"
    
    # Création des labels de scène
    scene_labels = []
    attributes_names = ['size', 'color', 'material', 'shape']
    
    for i in range(x_scene.shape[0]):
        if i in object_nodes:
            obj_idx = object_nodes.index(i)
            if obj_idx < len(scene_objects):
                scene_labels.append(f"OBJ{obj_idx+1}\n{scene_objects[obj_idx]}")
            else:
                scene_labels.append(f"OBJ{obj_idx+1}")
        else:
            sources = list(node_in_connections[i])
            if sources and sources[0] in object_nodes:
                source_obj_idx = object_nodes.index(sources[0])
                attr_count = 0
                for j in range(x_scene.shape[0]):
                    if j not in object_nodes and sources[0] in node_in_connections[j]:
                        if j == i:
                            break
                        attr_count += 1
                
                if source_obj_idx < len(scene_objects_full):
                    parts = scene_objects_full[source_obj_idx].split()
                    if attr_count < len(parts) and attr_count < len(attributes_names):
                        attr_name = attributes_names[attr_count]
                        attr_value = parts[attr_count]
                        scene_labels.append(f"{attr_name}\n{attr_value}")
                    else:
                        scene_labels.append(f"attr{i}")
                else:
                    scene_labels.append(f"attr{i}")
            else:
                scene_labels.append(f"N{i}")
    
    # ==========================
    # AFFICHAGE DES RÉSULTATS
    # ==========================
    
    print("\n" + "=" * 80)
    print(f"📌 QUESTION: {' '.join(question_words_all)}")
    print("=" * 80 + "\n")
    
    gnn_type = opt.gnn.lower()
    
    # 1. Graphe de question
    if 'gat' in gnn_type:
        gnn_q = model.seq2seq.gnn.psi_1
        # Vérifier si le modèle supporte return_attention (supposé vrai si GAT)
        visualize_gat_attention(
            gat_model=gnn_q,
            x=x_question,
            edge_index=edge_index_s_remapped,
            layer_idx=0,
            head_idx=0,
            node_labels=question_labels,
            edge_relations=None,
            vocab=vocab,
            title=f"GAT - Structure de la Question",
            threshold=0.0,
            figsize=(14, 12)
        )
    else:
        # Pour GIN ou autre GNN sans attention
        visualize_graph_structure(
            x=x_question,
            edge_index=edge_index_s_remapped,
            node_labels=question_labels,
            edge_relations=None,
            title="Structure du graphe - Question",
            figsize=(14, 12)
        )
    
    # 2. Image de la scène
    if 'val' in image_filename.lower():
        image_path = os.path.join('data', 'CLEVR_v1.0', 'images', 'val', image_filename)
    else:
        image_path = os.path.join('data', 'CLEVR_v1.0', 'images', 'train', image_filename)
    
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Scène: {image_filename}\n{len(scene_objects)} objets", 
                 fontsize=12, pad=15)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Image non trouvée: {image_path}")
    
    # 3. Graphe de scène
    if 'gat' in gnn_type:
        # Gestion du Two-Stream vs Single-Stream
        if opt.two_stream_gnn:
             gnn_t = model.seq2seq.gnn.psi_2
        else:
             gnn_t = model.seq2seq.gnn.psi_1
            
        visualize_gat_attention(
            gat_model=gnn_t,
            x=x_scene,
            edge_index=edge_index_t_remapped,
            layer_idx=0,
            head_idx=0,
            node_labels=scene_labels,
            edge_relations=edge_relations,
            vocab=vocab,
            title=f"GAT - Structure de la Scène",
            threshold=0.0,
            figsize=(16, 14)
        )
    else:
        # Pour GIN ou autre GNN sans attention
        visualize_graph_structure(
            x=x_scene,
            edge_index=edge_index_t_remapped,
            node_labels=scene_labels,
            edge_relations=edge_relations,
            title=f"Structure du graphe - Scène",
            figsize=(16, 14)
        )
    
    # 4. Génération du programme par le modèle
    # On utilise tout le batch et on extrait le programme pour l'échantillon voulu
    program_text = []
    answer_predicted = "N/A"
    
    try:
        model.seq2seq.eval()
        with torch.no_grad():
            # Utiliser le batch complet
            model.set_input(questions, programs, g_data_batch)
            all_programs = model.parse()
            predicted_program = all_programs[idx_to_extract]
        
        # Convertir le programme prédit en texte
        program_vocab = vocab.get('program_idx_to_token', vocab.get('answer_idx_to_token', {}))
        for token_idx in predicted_program.cpu().numpy():
            if token_idx in program_vocab:
                token = program_vocab[token_idx]
                if token in ['<NULL>', '<START>', '<END>']:
                    if token == '<END>':
                        break
                    continue
                program_text.append(token)
        
        # 5. Exécuter le programme pour obtenir la réponse
        if program_text and scene:
            try:
                from executors import get_executor
                executor = get_executor(opt)
                answer_predicted = executor.run(program_text, scene, 'program')
            except Exception as e:
                answer_predicted = f"Erreur execution: {str(e)[:50]}"
        else:
            if not program_text:
                program_text = ["<programme vide>"]
            answer_predicted = "Scène non disponible" if not scene else "N/A"
                
    except Exception as e:
        program_text = [f"Erreur: {str(e)[:50]}"]
        answer_predicted = "N/A"
        
    print("\n" + "=" * 80)
    print(f"🤖 PROGRAMME GÉNÉRÉ: {' '.join(program_text)}")
    print("-" * 40)
    print(f"🏁 RÉPONSE DU PROGRAMME: {answer_predicted}")
    print("=" * 80 + "\n")


def visualize_programs_and_predictions(model, val_loader, val_dataset, opt, num_examples=3, seed=0):
    """Visualise question, graphes, image et prédiction pour toutes les configs (GAT/GIN, two-stream, classification ou seq2seq)."""
    import random

    classification_mode = bool(getattr(opt, 'classification_mode', False)) or hasattr(model, 'classifier')
    gat_mode = 'gat' in str(getattr(opt, 'gnn', '')).lower()
    two_stream = bool(getattr(opt, 'two_stream_gnn', False))
    rng = random.Random(seed)

    # Charger questions et scènes
    output_dir = f"clevr_preprocessed_{opt.max_train_samples + opt.max_val_samples}"
    val_questions_path = os.path.join(output_dir, 'temp', f"val_questions_{opt.max_val_samples}.json")
    with open(val_questions_path, 'r') as f:
        val_questions = json.load(f).get('questions', [])

    with open(opt.clevr_val_scene_path, 'r') as f:
        scenes_data = json.load(f).get('scenes', [])

    vocab = val_dataset.vocab
    idx2answer = vocab.get('answer_idx_to_token', {})
    idx2program = vocab.get('program_idx_to_token', idx2answer)

    if not val_questions:
        print("⚠️ Aucun échantillon de validation chargé, rien à afficher.")
        return

    max_available = min(len(val_questions), len(val_dataset))
    if max_available == 0:
        print("⚠️ Dataset de validation vide, rien à afficher.")
        return

    num_to_show = min(num_examples, max_available)
    sample_indices = rng.sample(list(range(max_available)), num_to_show)
    sample_targets = set(sample_indices)

    # Executor uniquement pour le mode programme
    executor = None
    if not classification_mode:
        try:
            from executors import get_executor
            executor = get_executor(opt)
        except Exception as e:
            print(f"⚠️ Executor indisponible, les réponses seront non exécutées ({str(e)[:80]})")

    # Sélection des GNN pour visualiser l'attention/structure
    def _get_gnn_heads():
        if classification_mode:
            gnn_container = getattr(model, 'gnn', None)
        else:
            gnn_container = getattr(getattr(model, 'seq2seq', None), 'gnn', None)
        if gnn_container is None:
            return None, None
        q_gnn = getattr(gnn_container, 'psi_1', None)
        t_gnn = getattr(gnn_container, 'psi_2', None) if two_stream and getattr(gnn_container, 'psi_2', None) else getattr(gnn_container, 'psi_1', None)
        return q_gnn, t_gnn

    question_gnn, scene_gnn = _get_gnn_heads()

    print("\n" + "=" * 80)
    print(f"🎯 Visualisation XAI ({'classification' if classification_mode else 'programme'}) | GNN: {opt.gnn} | Two-stream: {two_stream}")
    print("=" * 80)

    shown = 0
    global_idx = 0

    for batch_id, batch in enumerate(val_loader):
        if shown >= num_to_show:
            break

        questions, programs, answers, image_idxs, g_data = batch
        BATCH_S, BATCH_T = g_data
        batch_size = questions.shape[0]

        # Prédictions (classification ou génération de programme)
        if classification_mode:
            model.eval()
            model.set_input(x=questions, g_data=g_data, answers=answers)
            with torch.no_grad():
                logits = model.forward()
                probs = torch.softmax(logits, dim=-1)
                pred_indices = torch.argmax(probs, dim=-1)
        else:
            model.seq2seq.eval()
            with torch.no_grad():
                model.set_input(questions, programs, g_data)
                generated_programs = model.parse()

        for pos_in_batch in range(batch_size):
            dataset_idx = batch_id * val_loader.batch_size + pos_in_batch
            global_idx = dataset_idx
            if dataset_idx not in sample_targets:
                continue

            shown += 1
            question_data = val_questions[dataset_idx]
            question_text = question_data.get('question', '').strip()
            true_answer = question_data.get('answer', '')
            image_filename = question_data.get('image_filename', '')
            image_index = int(question_data.get('image_index', image_idxs[pos_in_batch].item() if isinstance(image_idxs, torch.Tensor) else image_idxs[pos_in_batch]))

            # Extraire sous-graphes pour l'échantillon
            mask_s = BATCH_S.batch == pos_in_batch
            mask_t = BATCH_T.batch == pos_in_batch

            x_question = BATCH_S.x[mask_s]
            x_scene = BATCH_T.x[mask_t]

            edge_mask_s = mask_s[BATCH_S.edge_index[0]] & mask_s[BATCH_S.edge_index[1]]
            edge_index_s = BATCH_S.edge_index[:, edge_mask_s]

            edge_mask_t = mask_t[BATCH_T.edge_index[0]] & mask_t[BATCH_T.edge_index[1]]
            edge_index_t = BATCH_T.edge_index[:, edge_mask_t]

            node_mapping_s = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.where(mask_s)[0])}
            node_mapping_t = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.where(mask_t)[0])}

            edge_index_s_remapped = torch.tensor([
                [node_mapping_s[src.item()] for src in edge_index_s[0]],
                [node_mapping_s[tgt.item()] for tgt in edge_index_s[1]]
            ], dtype=torch.long)

            edge_index_t_remapped = torch.tensor([
                [node_mapping_t[src.item()] for src in edge_index_t[0]],
                [node_mapping_t[tgt.item()] for tgt in edge_index_t[1]]
            ], dtype=torch.long)

            # Labels de question
            question_tokens = questions[pos_in_batch].cpu().numpy()
            question_words_all = []
            for token_idx in question_tokens:
                if token_idx in vocab.get('question_idx_to_token', {}):
                    word = vocab['question_idx_to_token'][token_idx]
                    if word == '<END>':
                        break
                    if word not in ['<NULL>', '<START>']:
                        question_words_all.append(word)

            question_labels = []
            if len(question_words_all) <= x_question.shape[0]:
                for i in range(x_question.shape[0]):
                    question_labels.append(question_words_all[i] if i < len(question_words_all) else f"syntrel_{i}")
            else:
                words_per_node = len(question_words_all) / max(1, x_question.shape[0])
                for i in range(x_question.shape[0]):
                    start_idx = int(i * words_per_node)
                    end_idx = int((i + 1) * words_per_node)
                    node_words = question_words_all[start_idx:end_idx]
                    question_labels.append(' '.join(node_words) if node_words else f"Q{i}")

            # Récupérer la scène complète
            scene = next((s for s in scenes_data if s.get('image_index') == image_index), None)

            # Labels et relations de scène
            scene_objects = []
            scene_objects_full = []
            if scene:
                for obj in scene.get('objects', []):
                    size = obj.get('size', '')
                    color = obj.get('color', '')
                    shape = obj.get('shape', '')
                    material = obj.get('material', '')
                    scene_objects.append(f"{color[:3]}{shape[:3]}")
                    scene_objects_full.append(f"{size} {material} {color} {shape}")
            else:
                scene_objects = [f"O{i}" for i in range(x_scene.shape[0])]
                scene_objects_full = scene_objects

            temp_out_connections = defaultdict(set)
            for src, tgt in edge_index_t_remapped.t().tolist():
                temp_out_connections[src].add(tgt)
            object_nodes = [i for i in range(x_scene.shape[0]) if len(temp_out_connections[i]) > 0]

            obj_positions = {}
            if scene and scene.get('objects'):
                for obj_idx, obj in enumerate(scene['objects']):
                    if obj_idx < len(object_nodes):
                        obj_positions[object_nodes[obj_idx]] = obj.get('3d_coords', [0, 0, 0])

            edge_relations = {}
            for src, tgt in edge_index_t_remapped.t().tolist():
                if src in object_nodes and tgt in object_nodes:
                    if src in obj_positions and tgt in obj_positions:
                        edge_relations[(src, tgt)] = calculate_spatial_relation(obj_positions[src], obj_positions[tgt])
                    else:
                        edge_relations[(src, tgt)] = "spatial"
                elif src in object_nodes and tgt not in object_nodes:
                    attr_types = ['hasSize', 'hasColor', 'hasMaterial', 'hasShape']
                    obj_attrs = [t for s, t in edge_index_t_remapped.t().tolist() if s == src and t not in object_nodes]
                    attr_position = obj_attrs.index(tgt) if tgt in obj_attrs else 0
                    edge_relations[(src, tgt)] = attr_types[attr_position] if attr_position < len(attr_types) else "hasAttr"
                else:
                    edge_relations[(src, tgt)] = "other"

            node_in_connections = defaultdict(set)
            for src, tgt in edge_index_t_remapped.t().tolist():
                node_in_connections[tgt].add(src)

            scene_labels = []
            attributes_names = ['size', 'color', 'material', 'shape']
            for i in range(x_scene.shape[0]):
                if i in object_nodes:
                    obj_idx = object_nodes.index(i)
                    scene_labels.append(f"OBJ{obj_idx+1}\n{scene_objects[obj_idx]}" if obj_idx < len(scene_objects) else f"OBJ{obj_idx+1}")
                else:
                    sources = list(node_in_connections[i])
                    if sources and sources[0] in object_nodes:
                        source_obj_idx = object_nodes.index(sources[0])
                        attr_count = sum(1 for j in range(i) if j not in object_nodes and sources[0] in node_in_connections[j])
                        if source_obj_idx < len(scene_objects_full):
                            parts = scene_objects_full[source_obj_idx].split()
                            if attr_count < len(parts) and attr_count < len(attributes_names):
                                scene_labels.append(f"{attributes_names[attr_count]}\n{parts[attr_count]}")
                            else:
                                scene_labels.append(f"attr{i}")
                        else:
                            scene_labels.append(f"attr{i}")
                    else:
                        scene_labels.append(f"N{i}")

            # Affichage textuel
            print("\n" + "-" * 80)
            print(f"📌 Exemple {shown}/{num_to_show} | idx dataset: {dataset_idx}")
            print(f"❓ Question: {question_text}")
            print(f"✅ Réponse attendue: {true_answer}")

            # Graphe question
            if gat_mode and question_gnn is not None:
                visualize_gat_attention(
                    gat_model=question_gnn,
                    x=x_question,
                    edge_index=edge_index_s_remapped,
                    layer_idx=0,
                    head_idx=0,
                    node_labels=question_labels,
                    edge_relations=None,
                    vocab=vocab,
                    title="GAT - Question",
                    threshold=0.0,
                    figsize=(14, 12)
                )
            else:
                visualize_graph_structure(
                    x=x_question,
                    edge_index=edge_index_s_remapped,
                    node_labels=question_labels,
                    edge_relations=None,
                    title="Structure du graphe - Question",
                    figsize=(14, 12)
                )

            # Image de scène
            if 'val' in image_filename.lower():
                image_path = os.path.join('data', 'CLEVR_v1.0', 'images', 'val', image_filename)
            else:
                image_path = os.path.join('data', 'CLEVR_v1.0', 'images', 'train', image_filename)

            try:
                img = Image.open(image_path)
                plt.figure(figsize=(10, 7))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Scène: {image_filename}\n{len(scene_objects)} objets", fontsize=12, pad=15)
                plt.tight_layout()
                plt.show()
            except FileNotFoundError:
                print(f"⚠️ Image non trouvée: {image_path}")

            # Graphe scène
            if gat_mode and scene_gnn is not None:
                visualize_gat_attention(
                    gat_model=scene_gnn,
                    x=x_scene,
                    edge_index=edge_index_t_remapped,
                    layer_idx=0,
                    head_idx=0,
                    node_labels=scene_labels,
                    edge_relations=edge_relations,
                    vocab=vocab,
                    title="GAT - Scène",
                    threshold=0.0,
                    figsize=(16, 14)
                )
            else:
                visualize_graph_structure(
                    x=x_scene,
                    edge_index=edge_index_t_remapped,
                    node_labels=scene_labels,
                    edge_relations=edge_relations,
                    title="Structure du graphe - Scène",
                    figsize=(16, 14)
                )

            # Programme ou prédiction
            if classification_mode:
                pred_idx = int(pred_indices[pos_in_batch].item())
                predicted_answer = idx2answer.get(pred_idx, str(pred_idx))
                confidence = probs[pos_in_batch, pred_idx].item() if 'probs' in locals() else 0.0
                status = "✅" if predicted_answer.lower() == true_answer.lower() else "❌"
                print(f"🤖 Prédiction directe: {predicted_answer} (confiance: {confidence:.2%}) {status}")
            else:
                program_tokens = generated_programs[pos_in_batch].cpu().numpy()
                program_words = []
                for token in program_tokens:
                    token_int = int(token)
                    if token_int in idx2program:
                        word = idx2program[token_int]
                        if word == '<END>':
                            break
                        if word not in ['<NULL>', '<START>']:
                            program_words.append(word)

                program_text = " ".join(program_words) if program_words else "[PROGRAMME VIDE]"
                answer_predicted = "N/A"
                if executor and scene:
                    try:
                        answer_predicted = executor.run(program_words, scene, 'program')
                    except Exception as e:
                        answer_predicted = f"Erreur exec: {str(e)[:60]}"

                status = "✅" if str(answer_predicted).lower() == true_answer.lower() else "❌"
                print(f"🤖 Programme généré: {program_text}")
                print(f"🏁 Réponse exécutée: {answer_predicted} {status}")

            if shown >= num_to_show:
                break

    print("\n" + "=" * 80)
    print("✅ Visualisation terminée")


def visualize_sample_classifier(model, val_loader, val_dataset, opt, sample_idx=0):
    """
    Visualise un échantillon en mode CLASSIFICATION (sans programme).
    
    Args:
        model: Le modèle ClassifierModel
        val_loader: DataLoader de validation
        val_dataset: Dataset de validation
        opt: Options du modèle
        sample_idx: Index de l'échantillon dans le batch (0 par défaut)
    """
    # Chargement des données
    batch = next(iter(val_loader))
    questions, programs, answers, image_idxs, g_data_batch = batch
    BATCH_S, BATCH_T = g_data_batch
    
    idx_to_extract = sample_idx
    vocab = val_dataset.vocab
    
    # Mapping des réponses
    answer_idx_to_token = {v: k for k, v in vocab['answer_token_to_idx'].items()}
    
    # Extraction des graphes
    mask_s = BATCH_S.batch == idx_to_extract
    mask_t = BATCH_T.batch == idx_to_extract
    
    x_question = BATCH_S.x[mask_s]
    x_scene = BATCH_T.x[mask_t]
    
    edge_mask_s = mask_s[BATCH_S.edge_index[0]] & mask_s[BATCH_S.edge_index[1]]
    edge_index_s = BATCH_S.edge_index[:, edge_mask_s]
    
    edge_mask_t = mask_t[BATCH_T.edge_index[0]] & mask_t[BATCH_T.edge_index[1]]
    edge_index_t = BATCH_T.edge_index[:, edge_mask_t]
    
    # Renumérotation des nœuds
    node_mapping_s = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.where(mask_s)[0])}
    node_mapping_t = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.where(mask_t)[0])}
    
    edge_index_s_remapped = torch.tensor([
        [node_mapping_s[src.item()] for src in edge_index_s[0]],
        [node_mapping_s[tgt.item()] for tgt in edge_index_s[1]]
    ], dtype=torch.long)
    
    edge_index_t_remapped = torch.tensor([
        [node_mapping_t[src.item()] for src in edge_index_t[0]],
        [node_mapping_t[tgt.item()] for tgt in edge_index_t[1]]
    ], dtype=torch.long)
    
    # Métadonnées
    question_tokens = questions[idx_to_extract].cpu().numpy()
    question_words_all = []
    for token_idx in question_tokens:
        if token_idx in vocab['question_idx_to_token']:
            word = vocab['question_idx_to_token'][token_idx]
            if word == '<END>':
                break
            if word not in ['<NULL>', '<START>']:
                question_words_all.append(word)
    
    image_idx = image_idxs[idx_to_extract].item()
    
    OUTPUT_DIR = f'clevr_preprocessed_{opt.max_train_samples + opt.max_val_samples}'
    val_questions_path = os.path.join(OUTPUT_DIR, 'temp', f'val_questions_{opt.max_val_samples}.json')
    with open(val_questions_path, 'r') as f:
        val_questions_json = json.load(f)['questions']
    
    question_data = val_questions_json[idx_to_extract]
    image_filename = question_data['image_filename']
    true_answer = question_data['answer']  # Vraie réponse du JSON
    
    with open(opt.clevr_val_scene_path, 'r') as f:
        scenes_data = json.load(f)
    
    scene = None
    for s in scenes_data['scenes']:
        if s['image_index'] == image_idx:
            scene = s
            break
    
    # Labels pour la scène
    if scene:
        scene_objects = []
        scene_objects_full = []
        for obj in scene['objects']:
            size = obj.get('size', '')
            color = obj.get('color', '')
            shape = obj.get('shape', '')
            material = obj.get('material', '')
            
            label = f"{color[:3]}{shape[:3]}"
            scene_objects.append(label)
            
            full_label = f"{size} {material} {color} {shape}"
            scene_objects_full.append(full_label)
    else:
        scene_objects = [f"O{i}" for i in range(x_scene.shape[0])]
        scene_objects_full = scene_objects
    
    # Labels pour la question
    question_labels = []
    if len(question_words_all) <= x_question.shape[0]:
        for i in range(x_question.shape[0]):
            if i < len(question_words_all):
                question_labels.append(question_words_all[i])
            else:
                question_labels.append(f"syntrel_{i}")
    else:
        words_per_node = len(question_words_all) / x_question.shape[0]
        for i in range(x_question.shape[0]):
            start_idx = int(i * words_per_node)
            end_idx = int((i + 1) * words_per_node)
            node_words = question_words_all[start_idx:end_idx]
            if node_words:
                question_labels.append(' '.join(node_words))
            else:
                question_labels.append(f"Q{i}")
    
    # ==========================
    # AFFICHAGE DES RÉSULTATS
    # ==========================
    
    print("\n" + "=" * 80)
    print(f"📌 QUESTION: {' '.join(question_words_all)}")
    print(f"✅ VRAIE RÉPONSE: {true_answer}")
    print("=" * 80 + "\n")
    
    gnn_type = opt.gnn.lower()
    
    # 1. Graphe de question
    if 'gat' in gnn_type:
        gnn_q = model.gnn.psi_1  # En mode classifier, model.gnn = GraphMatcher
        visualize_gat_attention(
            gat_model=gnn_q,
            x=x_question,
            edge_index=edge_index_s_remapped,
            layer_idx=0,
            head_idx=0,
            node_labels=question_labels,
            edge_relations=None,
            vocab=vocab,
            title=f"GAT - Structure de la Question",
            threshold=0.0,
            figsize=(14, 12)
        )
    else:
        visualize_graph_structure(
            x=x_question,
            edge_index=edge_index_s_remapped,
            node_labels=question_labels,
            edge_relations=None,
            title="Structure du graphe - Question",
            figsize=(14, 12)
        )
    
    # 2. Image de la scène
    if 'val' in image_filename.lower():
        image_path = os.path.join('data', 'CLEVR_v1.0', 'images', 'val', image_filename)
    else:
        image_path = os.path.join('data', 'CLEVR_v1.0', 'images', 'train', image_filename)
    
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Scène: {image_filename}\n{len(scene_objects)} objets", 
                 fontsize=12, pad=15)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Image non trouvée: {image_path}")
    
    # 3. Graphe de scène (avec attention si GAT)
    # Analyse de la structure
    temp_out_connections = defaultdict(set)
    for idx, (src, tgt) in enumerate(edge_index_t_remapped.t().tolist()):
        temp_out_connections[src].add(tgt)
    
    object_nodes = [i for i in range(x_scene.shape[0]) if len(temp_out_connections[i]) > 0]
    
    obj_positions = {}
    if scene and len(scene['objects']) > 0:
        for obj_idx, obj in enumerate(scene['objects']):
            if obj_idx < len(object_nodes):
                node_idx = object_nodes[obj_idx]
                obj_positions[node_idx] = obj['3d_coords']
    
    edge_relations = {}
    for idx, (src, tgt) in enumerate(edge_index_t_remapped.t().tolist()):
        if src in object_nodes and tgt in object_nodes:
            if src in obj_positions and tgt in obj_positions:
                edge_relations[(src, tgt)] = calculate_spatial_relation(obj_positions[src], obj_positions[tgt])
            else:
                edge_relations[(src, tgt)] = "spatial"
        elif src in object_nodes and tgt not in object_nodes:
            attr_types = ['hasSize', 'hasColor', 'hasMaterial', 'hasShape']
            obj_attrs = [t for s, t in edge_index_t_remapped.t().tolist() 
                        if s == src and t not in object_nodes]
            attr_position = obj_attrs.index(tgt) if tgt in obj_attrs else 0
            if attr_position < len(attr_types):
                edge_relations[(src, tgt)] = attr_types[attr_position]
            else:
                edge_relations[(src, tgt)] = "hasAttr"
        else:
            edge_relations[(src, tgt)] = "other"
    
    # Scene labels
    scene_labels = []
    attributes_names = ['size', 'color', 'material', 'shape']
    node_in_connections = defaultdict(set)
    for src, tgt in edge_index_t_remapped.t().tolist():
        node_in_connections[tgt].add(src)
    
    for i in range(x_scene.shape[0]):
        if i in object_nodes:
            obj_idx = object_nodes.index(i)
            if obj_idx < len(scene_objects):
                scene_labels.append(f"OBJ{obj_idx+1}\n{scene_objects[obj_idx]}")
            else:
                scene_labels.append(f"OBJ{obj_idx+1}")
        else:
            sources = list(node_in_connections[i])
            if sources and sources[0] in object_nodes:
                source_obj_idx = object_nodes.index(sources[0])
                attr_count = sum(1 for j in range(i) if j not in object_nodes and sources[0] in node_in_connections[j])
                if source_obj_idx < len(scene_objects_full):
                    parts = scene_objects_full[source_obj_idx].split()
                    if attr_count < len(parts) and attr_count < len(attributes_names):
                        scene_labels.append(f"{attributes_names[attr_count]}\n{parts[attr_count]}")
                    else:
                        scene_labels.append(f"attr{i}")
                else:
                    scene_labels.append(f"attr{i}")
            else:
                scene_labels.append(f"N{i}")
    
    if 'gat' in gnn_type:
        if getattr(opt, 'two_stream_gnn', 0):
            gnn_t = model.gnn.psi_2
        else:
            gnn_t = model.gnn.psi_1
            
        visualize_gat_attention(
            gat_model=gnn_t,
            x=x_scene,
            edge_index=edge_index_t_remapped,
            layer_idx=0,
            head_idx=0,
            node_labels=scene_labels,
            edge_relations=edge_relations,
            vocab=vocab,
            title=f"GAT - Structure de la Scène",
            threshold=0.0,
            figsize=(16, 14)
        )
    else:
        visualize_graph_structure(
            x=x_scene,
            edge_index=edge_index_t_remapped,
            node_labels=scene_labels,
            edge_relations=edge_relations,
            title=f"Structure du graphe - Scène",
            figsize=(16, 14)
        )
    
    # 4. Prédiction DIRECTE par le classificateur (pas de programme!)
    predicted_answer = "N/A"
    confidence = 0.0
    
    try:
        model.eval()
        with torch.no_grad():
            # Set input avec les graphes du batch
            model.set_input(x=questions, g_data=g_data_batch, answers=answers)
            
            # Obtenir les logits
            logits = model.forward()
            probs = torch.softmax(logits, dim=-1)
            
            # Extraire la prédiction pour l'échantillon voulu
            pred_idx = torch.argmax(probs[idx_to_extract]).item()
            confidence = probs[idx_to_extract, pred_idx].item()
            predicted_answer = answer_idx_to_token.get(pred_idx, '<UNK>')
            
            # Top 3 prédictions
            top_probs, top_indices = torch.topk(probs[idx_to_extract], k=min(3, probs.shape[1]))
            top3 = [(answer_idx_to_token.get(idx.item(), '<UNK>'), prob.item()) 
                    for idx, prob in zip(top_indices, top_probs)]
                
    except Exception as e:
        predicted_answer = f"Erreur: {str(e)[:50]}"
        top3 = []
    
    # Affichage final
    is_correct = predicted_answer.lower() == true_answer.lower()
    status_emoji = "✅" if is_correct else "❌"
    
    print("\n" + "=" * 80)
    print(f"🎯 MODE CLASSIFICATION (End-to-End VQA)")
    print("-" * 40)
    print(f"❓ Question: {' '.join(question_words_all)}")
    print(f"✅ Vraie réponse: {true_answer}")
    print(f"🤖 Prédiction: {predicted_answer} (confiance: {confidence:.2%}) {status_emoji}")
    if top3:
        print("-" * 40)
        print("📊 Top 3 prédictions:")
        for i, (ans, prob) in enumerate(top3, 1):
            print(f"   {i}. {ans}: {prob:.2%}")
    print("=" * 80 + "\n")
