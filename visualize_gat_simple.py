# Script simplifié pour visualiser GAT avec recherche de question spatiale
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from collections import defaultdict

def find_spatial_question(questions, vocab, image_idxs):
    """Cherche une question contenant des mots spatiaux"""
    spatial_keywords = ['left', 'right', 'front', 'behind']
    
    for batch_idx in range(len(image_idxs)):
        q_tokens = questions[batch_idx].cpu().numpy()
        q_words = []
        for token_idx in q_tokens:
            if token_idx in vocab['question_idx_to_token']:
                word = vocab['question_idx_to_token'][token_idx]
                if word == '<END>':
                    break
                q_words.append(word.lower())
        
        if any(keyword in q_words for keyword in spatial_keywords):
            print(f"✅ Question spatiale trouvée: {' '.join(q_words)}")
            return batch_idx, q_words
    
    print("⚠️ Aucune question spatiale, utilisation de la première")
    return 0, None

def visualize_gat_simple(gat_model, x, edge_index, node_labels, edge_relations, title, threshold=0.0):
    """Visualisation simplifiée du GAT"""
    gat_model.eval()
    with torch.no_grad():
        out, attn_list = gat_model(x, edge_index, return_attention=True)
        edge_idx_layer, alpha = attn_list[0]
        attn_weights = alpha[:, 0].cpu().numpy()
    
    sources = edge_idx_layer[0].cpu().numpy()
    targets = edge_idx_layer[1].cpu().numpy()
    
    G = nx.DiGraph()
    for i in range(x.shape[0]):
        label = node_labels[i] if i < len(node_labels) else f"N{i}"
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
    
    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=9, ax=ax)
    
    if edges_to_draw:
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=widths, 
                              edge_color=colors, edge_cmap=plt.cm.Reds, 
                              edge_vmin=0, edge_vmax=1,
                              arrowstyle='->', arrowsize=20,
                              connectionstyle='arc3,rad=0.1', ax=ax)
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict,
                                     font_size=8, font_color='darkred', ax=ax)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Attention", shrink=0.8)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"✅ {x.shape[0]} nœuds, {len(edges_to_draw)} arêtes")
    
    # Statistiques des relations
    if edge_relations:
        relation_counts = defaultdict(int)
        for (s, t), rel_name in edge_relations.items():
            if (s, t) in edge_labels_dict and isinstance(rel_name, str) and not rel_name.startswith('rel'):
                relation_counts[rel_name] += 1
        
        if relation_counts:
            print("🔗 Relations spatiales:")
            for rel_name, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
                print(f"   • {rel_name}: {count}")
