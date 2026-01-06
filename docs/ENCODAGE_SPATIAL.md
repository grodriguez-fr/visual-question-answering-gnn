# 🧠 Architecture d'Encodage Spatial dans MGN

## 📋 Résumé Exécutif

**Découverte clé :** Le système MGN n'encode PAS les relations spatiales comme des arêtes symboliques dans le graphe. Au lieu de cela, il fournit directement les **coordonnées 3D brutes** (x, y, z) dans les features des nœuds et laisse le GNN apprendre à les interpréter.

---

## 🏗️ Architecture Complète

### Structure des Données

#### Graphe de Scène (`G_t`)

**Nœuds :**

- Objets principaux (cube, sphère, cylindre, etc.)
- Nœuds d'attributs (couleur, taille, matériau, forme)

**Features de Nœud (99D) :**

```
x_node = [embedding_96D, x_3D, y_3D, z_3D]
         └──────┬──────┘  └──────┬────────┘
          Sémantique     Positions 3D brutes
          (spaCy)        (depuis JSON CLEVR)
```

**Arêtes :**

- `hasColor` : Objet → Couleur
- `hasSize` : Objet → Taille
- `hasMaterial` : Objet → Matériau
- `hasShape` : Objet → Forme

**❌ PAS d'arêtes :**

- `left(obj1, obj2)`
- `right(obj1, obj2)`
- `front(obj1, obj2)`
- `behind(obj1, obj2)`

#### Graphe de Question (`G_q`)

**Nœuds :**

- Mots de la question (analyse syntaxique)
- Relations syntaxiques

**Features de Nœud (96D) :**

```
x_node = [embedding_96D]
         └──────┬──────┘
          Sémantique
          (spaCy)
```

---

## 🔄 Double Usage des Positions 3D

### 1️⃣ Phase d'Entraînement : GNN

**Input :**

```python
x_scene = torch.tensor([
    [emb_96D..., x_pos, y_pos, z_pos],  # Objet 1
    [emb_96D..., x_pos, y_pos, z_pos],  # Objet 2
    # ...
])
```

**Ce que le GNN doit apprendre :**

- Comparer `x_1 < x_2` → "obj1 est à gauche de obj2"
- Comparer `y_1 < y_2` → "obj1 est derrière obj2"
- Calculer distances euclidiennes pour "proche/loin"
- Raisonnement contextuel (importance relative des positions)

**Mécanisme :**

```
GAT Layer 1: Agrège features de voisins avec attention
             ↓
             Apprend à pondérer les positions 3D
             ↓
GAT Layer 2: Raffine les représentations
             ↓
             Encode implicitement les relations spatiales
```

### 2️⃣ Phase de Validation : Executor

**Code Python :**

```python
# executors/clevr_executor.py

def filter_left(scene, target_obj):
    """Filtre objets à gauche de target_obj"""
    return [obj for obj in scene
            if obj['3d_coords'][0] < target_obj['3d_coords'][0]]

def filter_right(scene, target_obj):
    """Filtre objets à droite de target_obj"""
    return [obj for obj in scene
            if obj['3d_coords'][0] > target_obj['3d_coords'][0]]

def filter_front(scene, target_obj):
    """Filtre objets devant target_obj"""
    return [obj for obj in scene
            if obj['3d_coords'][1] > target_obj['3d_coords'][1]]

def filter_behind(scene, target_obj):
    """Filtre objets derrière target_obj"""
    return [obj for obj in scene
            if obj['3d_coords'][1] < target_obj['3d_coords'][1]]
```

**Rôle :**

- Recalcule les relations spatiales depuis le JSON de scène
- Vérifie la validité des programmes générés par le modèle
- Accès à la vérité terrain (ground truth parfaite)

---

## 💾 Fichiers Précalculés (.pt)

### Contenu des Fichiers

```python
# clevr_train_questions_directed_pairdata.pt
{
    'data_s_list': [  # Questions
        ClevrData(
            x=torch.tensor([...]),        # [N_nodes, 96]
            edge_index=torch.tensor([...]),  # [2, N_edges]
            edge_attr=torch.tensor([...])    # [N_edges, 96]
        ),
        # ...
    ],
    'data_t_list': [  # Scènes
        ClevrData(
            x=torch.tensor([...]),        # [N_nodes, 99] ← 96D + 3D!
            edge_index=torch.tensor([...]),  # [2, N_edges]
            edge_attr=torch.tensor([...])    # [N_edges, 99]
        ),
        # ...
    ]
}
```

### ✅ Ce qui est sauvegardé

- Tenseurs PyTorch (x, edge_index, edge_attr)
- Embeddings sémantiques (96D)
- **Positions 3D brutes** dans x[:,96:99]
- Structure du graphe (nœuds, arêtes)

### ❌ Ce qui N'est PAS sauvegardé

- Graphes NetworkX avec métadonnées
- Noms textuels des relations ("left", "right")
- Labels textuels des nœuds
- Parsing syntaxique (Doc spaCy)

---

## 🎯 Implications

### Avantages ✅

1. **Flexibilité :** Le GNN peut apprendre des relations contextuelles

   - "proche" peut dépendre de la disposition globale
   - Relations continues au lieu de discrètes
   - Adaptation au contexte de la question

2. **Généralisation :** Pas limité à un ensemble fixe de relations

   - Peut apprendre des notions spatiales complexes
   - Transfert potentiel à d'autres domaines

3. **Approche End-to-End :** Tout appris depuis les données brutes
   - Pas de feature engineering manuel
   - Le réseau optimise l'utilisation des coordonnées

### Défis ⚠️

1. **Apprentissage Complexe :** Le GNN doit apprendre les comparaisons géométriques

   - Nécessite plus de données d'entraînement
   - Convergence potentiellement plus lente
   - Risque de sur-apprentissage

2. **Interprétabilité :** Difficile de comprendre comment le réseau utilise (x,y,z)

   - Les poids GAT montrent l'attention mais pas le calcul spatial
   - Boîte noire pour les comparaisons numériques

3. **Visualisation :** Impossible de "décoder" les embeddings vers du texte
   - Pour visualiser, doit recalculer les relations depuis le JSON
   - Nécessite `calculate_spatial_relation(pos1, pos2)`

---

## 🔍 Visualisation et Debugging

### Problème Rencontré

Lors de la visualisation GAT, les labels d'arêtes ne montraient pas les relations spatiales attendues.

**Cause :** Les relations spatiales ne sont PAS stockées dans edge_attr comme du texte, mais comme des embeddings 96D.

### Solution

Recalculer les relations spatiales pour la visualisation :

```python
def calculate_spatial_relation(obj1_pos, obj2_pos):
    """Calcule la relation spatiale entre deux objets"""
    dx = obj2_pos[0] - obj1_pos[0]  # x: left/right
    dy = obj2_pos[1] - obj1_pos[1]  # y: behind/front

    # Déterminer la relation dominante
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    else:
        return "behind" if dy < 0 else "front"

# Utilisation dans la visualisation
edge_relations = {}
for src, tgt in edge_list:
    if src in obj_positions and tgt in obj_positions:
        edge_relations[(src, tgt)] = calculate_spatial_relation(
            obj_positions[src],
            obj_positions[tgt]
        )
```

---

## 📚 Références Code

### Fichiers Clés

1. **Encodage des positions :**

   - `clevr-parser/clevr_parser/backends/torch_embedder.py:375-405`
   - Fonction `get_node_feature_matrix()` ajoute positions 3D

2. **Utilisation par le GNN :**

   - `models/gnn.py` : Couches GAT/GCN/GIN
   - Input : x avec 99D (96 + 3)

3. **Executor spatial :**

   - `executors/clevr_executor.py`
   - Fonctions `filter_left`, `filter_right`, etc.

4. **Précalcul des graphes :**
   - `tools/sample_questions_mgn.py:213-244`
   - Crée ClevrData avec embeddings + positions

---

## 🎓 Conclusion

L'architecture MGN adopte une approche **implicite** du raisonnement spatial : elle fournit les coordonnées 3D brutes et laisse le réseau apprendre à les interpréter de manière contextuelle. Cette approche contraste avec une représentation symbolique explicite et offre plus de flexibilité au prix d'une complexité d'apprentissage accrue.

**Message clé :** Les positions spatiales ne sont pas "cachées" dans les embeddings - elles sont directement accessibles dans les 3 dernières dimensions des features de nœud ! Le défi est d'apprendre à les utiliser efficacement.
