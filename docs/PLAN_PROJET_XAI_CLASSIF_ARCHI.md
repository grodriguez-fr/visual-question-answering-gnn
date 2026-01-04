# 🚀 Plan de Projet : XAI, Classification et Architecture Multimodale

Ce document définit les trois axes principaux de développement pour l'amélioration du projet Visual Question Answering avec GNN. Il servira de guideline pour l'implémentation et le suivi des résultats.

---

## 1. Axe XAI : Explicabilité par Graph Attention Networks (GAT)

**Introduction** :  
L'objectif est de rendre le modèle transparent en remplaçant les mécanismes de convolution de graphe standard par des mécanismes d'attention (GAT), permettant de visualiser quelles parties du graphe de scène et du graphe de question sont déterminantes pour la réponse.

### 🧐 Critique et Justification

Le papier original utilise des GNNs classiques (GCN ou variantes simples) qui agrègent l'information de manière uniforme ou selon des poids fixes. Cela rend le modèle "boîte noire" : on ne sait pas si le modèle regarde le bon objet dans la scène ou le bon mot dans la question pour générer le programme. L'absence de mécanismes d'attention explicites empêche de vérifier si le raisonnement est aligné avec l'intuition humaine.

### 💡 Solution Proposée

Nous allons implémenter une couche **Graph Attention Network (GAT)**. Contrairement aux GCNs, les GATs apprennent des poids d'attention $\alpha_{ij}$ pour chaque arête, indiquant l'importance du voisin $j$ pour le nœud $i$.

**Implémentation technique :**

- Remplacer les couches de convolution de graphe actuelles par des couches `GATConv` (PyTorch Geometric ou implémentation custom).
- Extraire les coefficients d'attention lors de la passe avant (forward pass).
- Créer des outils de visualisation pour superposer ces poids sur les graphes originaux.

### 📊 Résultats et Implémentation (Placeholders)

_À implémenter :_

```python
# TODO: Modifier le modèle GNN pour retourner les poids d'attention
# class GATModel(nn.Module):
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x, attention_weights = self.gat_conv(x, edge_index, return_attention_weights=True)
#         return x, attention_weights
```

**Métriques attendues :**

- [ ] Cartes de chaleur (Heatmaps) montrant les nœuds les plus activés pour une question donnée.
- [ ] Corrélation entre l'attention portée sur un objet et sa mention dans la question.
- [ ] Exemple qualitatif : Pour la question "Quelle est la couleur du cube ?", le modèle doit porter une attention forte sur le nœud "Cube" dans le graphe de scène.

---

## 2. Axe Classification : VQA End-to-End

**Introduction** :  
Nous allons implémenter une véritable tâche de classification pour répondre aux questions (VQA), au lieu de se limiter à la génération de programmes, afin de forcer le modèle à apprendre des représentations multimodales riches.

### 🧐 Critique et Justification

Le code actuel se concentre quasi-exclusivement sur la génération de programmes (Task 2 du papier), laissant l'exécution de ce programme à un moteur symbolique externe qui a accès à la vérité terrain (la scène parfaite). Cela contourne la difficulté principale du VQA : comprendre visuellement la scène pour répondre. Le papier mentionnait une tâche de classification (Task 1) qui n'a jamais été implémentée dans le code public. Sans classification, le GNN n'est pas obligé de compresser toute l'information nécessaire à la réponse dans ses embeddings.

### 💡 Solution Proposée

Ajouter une tête de classification (MLP) en sortie du module de fusion multimodal. Le modèle devra prédire directement la réponse (ex: "blue", "yes", "3") à partir des embeddings fusionnés des graphes de question et de scène.

**Implémentation technique :**

- Créer un vocabulaire de réponses possibles (couleurs, formes, nombres, oui/non).
- Ajouter une tête de prédiction `Classifier(hidden_dim, num_answers)`.
- Entraîner avec une Cross-Entropy Loss sur les réponses réelles.

### 📊 Résultats et Implémentation (Placeholders)

_À implémenter :_

```python
# TODO: Ajouter la tête de classification dans seq2seq_gnn.py ou similaire
# self.classifier = nn.Sequential(
#     nn.Linear(hidden_dim, hidden_dim // 2),
#     nn.ReLU(),
#     nn.Linear(hidden_dim // 2, num_answers)
# )
```

**Tableau de résultats (Placeholder) :**

| Modèle                            | Accuracy (Train) | Accuracy (Val) |
| :-------------------------------- | :--------------: | :------------: |
| Baseline (Programme seulement)    |       N/A        |      N/A       |
| **Notre Modèle (Classification)** |      _TBD_       |     _TBD_      |

---

## 3. Axe Architecture Séparée : Disjoint Scene & Question Graphs

**Introduction** :  
Nous proposons de séparer architecturalement le traitement du graphe de scène et du graphe de question avant leur fusion, pour respecter la sémantique distincte de ces deux modalités.

### 🧐 Critique et Justification

L'approche actuelle (ou celle suggérée par certaines implémentations naïves de "graphes multimodaux") tend à connecter tous les nœuds de la question à tous les nœuds de la scène dans un seul "super-graphe", ou à les traiter de manière trop entremêlée dès le début. Cela crée du bruit et empêche le modèle de structurer d'abord sa compréhension de la scène (visuel) et de la question (linguistique) indépendamment avant de les croiser. Une véritable approche multimodale doit encoder chaque modalité spécifiquement avant la fusion.

### 💡 Solution Proposée

Adopter une architecture à deux branches (Two-Stream) qui fusionnent ensuite :

1.  **Branche Scène** : Un GNN (GAT) qui traite uniquement les objets et leurs relations spatiales.
2.  **Branche Question** : Un GNN (ou LSTM/Transformer) qui traite la structure syntaxique/sémantique de la question.
3.  **Module de Fusion** : Un mécanisme d'attention croisée (Cross-Attention) ou un "Graph Matcher" qui lie les nœuds de la question aux nœuds de la scène pertinents.

### 📊 Résultats et Implémentation (Placeholders)

_Schéma d'architecture visé :_
`Scene Graph -> GAT_Scene -> Scene_Embeddings`
`Question Graph -> GAT_Question -> Question_Embeddings`
`[Scene_Embeddings, Question_Embeddings] -> Cross_Attention -> Joint_Representation`

_À implémenter :_

```python
# TODO: Refactoriser le modèle pour avoir deux encodeurs distincts
# scene_emb = self.scene_gnn(scene_graph)
# question_emb = self.question_gnn(question_graph)
# fused = self.cross_attention(scene_emb, question_emb)
```

**Hypothèse de performance :**

- Cette séparation devrait améliorer la généralisation sur des questions complexes nécessitant un raisonnement en plusieurs étapes, car la structure de la question sera mieux préservée.
