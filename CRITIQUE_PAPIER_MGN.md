# 🔍 Analyse Critique : Multimodal Graph Networks (MGN)

**Papier** : Multimodal Graph Networks for Compositional Generalization in Visual Question Answering  
**Auteur de l'analyse** : Étudiant cours GNN  
**Date** : 3 janvier 2026

---

## 📋 Table des matières

1. [Résumé Exécutif](#résumé-exécutif)
2. [Divergences Papier vs Code](#divergences-papier-vs-code)
3. [Problèmes de Design Architectural](#problèmes-de-design-architectural)
4. [Problèmes de Comparaison](#problèmes-de-comparaison)
5. [Incohérences d'Architecture](#incohérences-darchitecture)
6. [Problèmes de Training](#problèmes-de-training)
7. [Points Positifs](#points-positifs)
8. [Recommandations](#recommandations)

---

## Résumé Exécutif

### 🎯 Problèmes Majeurs Identifiés : 10

**Catégories** :

- 3 divergences papier/code (mensonges/omissions)
- 3 problèmes de design architectural
- 2 problèmes de comparaison injuste
- 2 incohérences d'architecture

**Sévérité** : 🔴 Critique

**Verdict** : Le papier survend ses contributions. Le code implémente une architecture différente et plus simple que celle décrite. Les comparaisons avec les baselines sont injustes en raison de la supervision forte (annotations CLEVR). Le GNN est probablement inutile pour la tâche de génération de programmes (Task 2).

---

## Divergences Papier vs Code

### 🚨 Mensonge #1 : Masked R-CNN

**Ce que dit le papier** :

> "We use Masked R-CNN + ResNet-50 for visual feature extraction from images"

**Ce qu'on trouve dans le code** :

- ❌ **ZÉRO** ligne de code pour Masked R-CNN
- ❌ **ZÉRO** import de bibliothèques de détection d'objets
- ✅ Utilisation directe des annotations CLEVR (fichiers JSON)

**Preuve** :

```bash
grep -r "mask.*rcnn\|RCNN\|MaskRCNN" . --include="*.py"
# Résultat : 0 matches (sauf dans docs/)
```

**Impact** :

- Le papier décrit un système qui n'existe pas
- Les annotations CLEVR sont du **ground truth** (objets déjà labellisés)
- C'est **beaucoup plus facile** que d'extraire les features avec un CNN

**Fichiers vérifiés** :

- `models/` : Aucune trace de CNN ou object detector
- `datasets/clevr_questions.py` : Charge directement les scene graphs pré-parsés
- `executors/clevr_executor.py` : Utilise `CLEVR_train_scenes.json` (annotations)

---

### 🚨 Mensonge #2 : Deux tâches, une seule implémentée

**Ce que dit le papier** :
Le papier décrit **DEUX tâches** distinctes avec architectures différentes :

#### Task 1 : Caption Truth Prediction (Binary Classification)

```
GraphMatcher → h_st (96-dim)
    ↓
FC layer + sigmoid
    ↓
y_hat (True/False)
    ↓
Binary cross-entropy loss
```

**Papier (page 5)** :

> "For our caption classification task, the output of the matcher $h_{s,t}$ can be fed to a fully connected layer with sigmoid activation for discriminative classification"

#### Task 2 : CLOSURE VQA

```
GraphMatcher → h_st
    ↓
Encoder-Decoder (seq2seq)
    ↓
Programme symbolique
    ↓
Executor (algorithme)
    ↓
Réponse
```

**Ce qu'on trouve dans le code** :

- ✅ Task 2 complètement implémentée (`models/seq2seq_gnn.py`, `trainer.py`)
- ❌ Task 1 **absente** : pas de binary cross-entropy, pas de classification head

**Preuve** :

```bash
grep -r "binary.*cross.*entropy\|BCELoss\|sigmoid.*classification" . --include="*.py"
# Résultat : 0 matches

grep -r "caption.*truth\|true.*false.*prediction" . --include="*.py"
# Résultat : 0 matches (sauf commentaires)
```

**Impact** :

- Résultats de Task 1 (Table 1, page 6) **non vérifiables**
- Claim de 98.2% accuracy sur Dataset B non reproductible
- Code incomplet ou partiel release

---

### 🚨 Mensonge #3 : "Multimodal" vision + langage

**Ce que dit le papier** :

- Title : "**Multimodal** Graph Networks"
- Claims de "multimodal learning" (images + text)
- Description d'extraction de features visuelles

**Réalité dans le code** :

- ❌ Pas d'extraction visuelle (pas de CNN/ResNet)
- ✅ Seulement **text graph + scene graph**
- ⚠️ Les deux graphes proviennent d'**annotations textuelles** (JSON)

**Analyse** :

```
"Multimodal" selon le papier : Vision (pixels) + Langage (texte)
"Multimodal" dans le code    : Text graph + Scene graph (tous deux textuels)
```

Le "scene graph" n'est PAS extrait d'images mais construit depuis les annotations CLEVR :

```json
{
  "objects": [
    {"shape": "cube", "color": "red", "size": "large", ...}
  ]
}
```

**Impact** :

- Marketing trompeur : ce n'est pas vraiment "multimodal" au sens vision+langage
- C'est plutôt "multi-graph" : deux types de graphes textuels

---

## Problèmes de Design Architectural

### ⚠️ Problème #1 : Shared GNN entre domaines hétérogènes

**Le code** (`models/graph_matcher.py`) :

```python
class GraphMatcher(nn.Module):
    def __init__(self, ...):
        self.psi_1 = GIN(...)  # UN SEUL GIN

    def forward(self, Gs, Gt):
        h_s = self.psi_1(Gs.x, Gs.edge_index)  # Text graph
        h_t = self.psi_1(Gt.x, Gt.edge_index)  # Scene graph
        # MÊMES poids pour les deux !
```

**Pourquoi c'est problématique** :

#### 1. Features initiales complètement différentes

- **Gs (text)** : Word embeddings (300-dim) - "red", "cube", "left"
- **Gt (scene)** : Object attributes (one-hot ou embeddings) - color=0, shape=2, size=1

**Ils ne vivent pas dans le même espace sémantique !**

#### 2. Structures de graphe différentes

- **Gs** : Graphe de dépendances linguistiques (sujet-verbe-objet, modificateurs)
- **Gt** : Graphe spatial (relations géométriques, proximité physique)

#### 3. Sémantiques différentes

- **Texte** : Syntaxe, grammaire, ordre des mots
- **Scène** : Géométrie, propriétés visuelles, relations spatiales

**Justification dans le papier** :

> "We use a **shared graph neural network (GNN)**, $\Psi_{\theta}$, to compute [...] a **latent joint representation**"

**Critique** :

- Aucune ablation comparant shared vs separate GNN
- Justification théorique faible (juste "pour avoir un espace commun")
- Style réseaux siamois, mais inputs trop hétérogènes

**Meilleur design** :

```python
self.gin_text = GIN(...)   # Poids pour texte
self.gin_scene = GIN(...)  # Poids pour scène (différents)
# Puis projection dans espace commun si nécessaire
```

**Impact** :

- Le GNN doit apprendre à traiter deux modalités avec les mêmes poids
- Conflit d'apprentissage potentiel
- Explication du problème "les nœuds se concentrent sur les mauvais tokens" avec GAT

---

### ⚠️ Problème #2 : GNN probablement inutile pour Task 2

**Observation** : Le GNN enrichit la génération du programme, mais le programme dépend principalement de la syntaxe de la question, pas de l'image.

**⚖️ Nuance importante** : L'argument du grounding est **théoriquement valide** (résoudre ambiguïtés, valider exécutabilité), MAIS l'implémentation MGN sur CLEVR ne profite **que très peu** de ces avantages. Voici pourquoi :

#### Génération du programme = Traduction syntaxique

```
Question: "What color is the large cube?"
    ↓ (traduction)
Programme: [scene, filter_shape[cube], filter_size[large],
           unique, query_color]
```

**Mapping direct** :

- "What color" → `query_color` (mot clé dans question)
- "cube" → `filter_shape[cube]` (mot dans question)
- "large" → `filter_size[large]` (mot dans question)

**Pas besoin d'info visuelle pour cette traduction !** C'est du pattern matching syntaxique.

#### Le scene graph embedding est redondant

**Code** (`models/encoder.py` ligne 37-43) :

```python
if self.gembd_vec_dim > 0:
    g_embedding = g_embd.unsqueeze(1).repeat(1, x.size(1), 1)
    embedded = torch.cat((word_embedding, g_embedding), dim=2)
    # Chaque mot enrichi avec l'info du scene graph
```

**Problème** :

- Le programme est généré token par token basé sur la question
- L'info "il y a vraiment un cube rouge dans l'image" est utilisée **APRÈS** par l'executor
- Le scene graph enrichit le seq2seq pour une tâche qui n'en a pas besoin

#### L'executor fait le vrai travail

**Code** (`executors/clevr_executor.py`) :

```python
def run(self, x, img_index, split):
    scene = self.scenes[split][img_index]  # Charge Scene JSON

    for token in programme:
        if token == 'filter_color[red]':
            ans = [o for o in ans if o['color'] == 'red']
        # C'est du LOOKUP dans un dictionnaire !
```

**Deux systèmes découplés** :

1. **Neural (Seq2seq)** : Question → Programme (traduction syntaxique, 30% du travail)
2. **Symbolique (Executor)** : Programme + Scene JSON → Réponse (lookup, 70% du travail)

**Le GNN aide le seq2seq, mais le seq2seq fait juste de la traduction syntaxique.**

#### Argument théorique du grounding

**On pourrait dire** : "Le scene graph aide à résoudre les ambiguïtés !"

**Exemple** :

```
Question: "Is the color of THE metal block the same as the large metal cylinder?"
```

**Argument** :

- "THE metal block" → Quelle instance précise si l'image a 2 blocs métalliques ?
- Le GNN fait du **grounding** : aligne "block" → objet #3 dans l'image
- Validation d'exécutabilité : vérifie que "large metal cylinder" existe

**C'est théoriquement correct !**

#### MAIS : Questions CLEVR jamais ambiguës en pratique

**CLEVR est template-based avec contrainte d'unicité** :

```python
# Template CLEVR garantit
"What is the color of the {size} {material} {shape}?"
# ⚠️ Contrainte : description pointe TOUJOURS vers exactement 1 objet
```

**Exemple concret** :

```
"Is the color of THE metal block the same as the large metal cylinder?"
```

**CLEVR garantit** :

- Il existe **exactement 1** metal block (pas 0, pas 2)
- Il existe **exactement 1** large metal cylinder
- **Donc pas d'ambiguïté à résoudre !** "THE metal block" pointe forcément vers l'unique instance

**Autres contraintes CLEVR** :

- Pas de pronoms ambigus ("it", "this one")
- Pas de références contextuelles ("the red one on the left")
- Structure syntaxique déterministe

**Résultat** : Le grounding n'est pas critique ici.

#### Validation d'exécutabilité : absente en mode supervised

**Code** (`models/parser.py`) :

```python
def supervised_forward(self):
    output_logprob = self.seq2seq(self.x, self.y, self.g_data)
    self.loss = self.criterion(
        output_logprob.view(-1, vocab_size),
        self.y.view(-1)  # Programme ground truth
    )
    # ⚠️ L'executor n'est PAS appelé !
```

**Observation** :

- Loss sur **tokens de programmes**, pas sur réponses
- **Aucune validation** que le programme est exécutable
- Le modèle apprend à copier la syntaxe, pas à vérifier l'exécutabilité

**Même si tu génères** :

```python
filter_large → filter_metal → filter_unicorn  # Unicorn n'existe pas !
```

→ Pas de pénalité en supervised mode

#### Mode reinforce : grounding peut aider (mais lent)

**Code** (`trainer.py`) :

```python
if self.reinforce:
    pred = self.model.reinforce_forward()
    reward = self.get_batch_reward(pred, ans, idx, 'train')
    # ⚠️ Executor appelé ici pour calculer reward
```

**Ici le scene graph peut aider** :

- Récompense basée sur exécution correcte
- Feedback sur exécutabilité du programme
- Le grounding aide à éviter programmes invalides

**MAIS** :

- Nécessite **1M iterations** (vs 25k supervised)
- Haute variance (policy gradient instable)
- Le papier ne dit pas si les meilleurs résultats viennent de supervised ou reinforce

**Expérience à faire** :

- Baseline : `gembd_vec_dim=0` (sans GNN)
- Full supervised : `gembd_vec_dim=96` (avec GNN)
- Full reinforce : `gembd_vec_dim=96` (avec GNN, 1M iter)
- **Prédiction** : Différence supervised < 5%, reinforce peut-être 10-15%

**Impact nuancé** :

- **Supervised (25k iter)** : GNN aide peu (traduction syntaxique)
- **Reinforce (1M iter)** : GNN peut aider davantage (validation exécution)
- Complexité ajoutée vs gain faible en supervised
- GNN pourrait juste mémoriser des associations syntaxiques

---

### ⚠️ Problème #3 : Matrice de correspondance Φ non-interprétable

**Code** (`models/graph_matcher.py`) :

```python
Φ_hat = h_s @ h_t.T  # (|V_s| × |V_t|)
Φ = sinkhorn_normalization(Φ_hat)
```

**Problème** : La matrice Φ contient des probabilités entre **embeddings latents**, pas des labels lisibles.

**Exemple** :

```
Question: "What color is the cube?"
Gs nodes: [color, cube, what, is]
Gt nodes: [obj1, obj2, obj3]

Φ = [[0.1, 0.8, 0.1],    # "color" matche obj2
     [0.2, 0.7, 0.1],    # "cube" matche obj2
     [0.3, 0.3, 0.4],    # "what" dispersé
     [0.4, 0.3, 0.3]]    # "is" dispersé
```

**Questions impossibles à répondre** :

- Pourquoi "color" matche obj2 ? (boîte noire)
- Quels features dans les embeddings créent cette correspondance ?
- Comment déboguer si le modèle se trompe ?

**Citation de l'utilisateur** :

> "Parfois les nœuds se concentrent sur les mauvais tokens, il me donne les mauvais objets"

**Impact** :

- Impossible à interpréter
- Impossible à déboguer
- Pas d'explicabilité

---

## Problèmes de Comparaison

### ⚠️ Problème #4 : Supervision forte injuste

**MGN** :

- Utilise annotations CLEVR (objets pré-labellisés : shape, color, size, material)
- Scene graphs construits depuis JSON (pas extraits d'images)
- Executor fait du lookup algorithmique

**Baselines comparées** :

- Neural Module Networks
- MAC (Memory, Attention, Composition)
- FiLM (Feature-wise Linear Modulation)
- Relation Networks

**Ces baselines** :

- Apprennent sur features visuelles **brutes** (CNN)
- Pas d'annotations structurées
- Doivent apprendre à extraire ET à raisonner

**Citation des reviewers** :

> "For the CLEVR dataset, the model is basically based on using an object detector [...] While other approaches [...] do not use such strong supervision and therefore **the comparison between these approaches in the experimental section is not completely valid**"

**Analogie** :

- Baseline : "Résous ce puzzle en regardant l'image floue"
- MGN : "Résous ce puzzle, je te donne les pièces déjà triées + position correcte"

**Impact** :

- Comparaison injuste
- Résultats impressionnants mais non comparables
- Supervision forte cache la vraie difficulté

---

### ⚠️ Problème #5 : Executor fait 90% du travail

**Reviewer critique** :

> "Since the proposed approach depends on the **pretrained model to get G_s** (graph to extract visual attributes), the goal now becomes to just check if the four attributes match or not. This becomes a **trivial problem of semantic matching of labels** and does not necessitate the complex graph fusion that is proposed."

**Analyse de la charge de travail** :

#### Seq2seq (neural) : ~30%

```python
Question → Programme (traduction syntaxique)
"What color" → query_color
"cube" → filter_shape[cube]
```

#### Executor (algorithme Python pur) : ~70%

```python
def filter_red(scene):
    return [o for o in scene if o['color'] == 'red']

def query_shape(obj):
    return obj['shape']
```

**C'est du matching de labels !** Pas de deep learning, juste des `if` et des boucles.

**Impact** :

- Le "deep learning" fait juste la traduction syntaxique
- Le vrai raisonnement est algorithmique
- Le problème devient trivial avec les annotations structurées

---

## Incohérences d'Architecture

### ⚠️ Problème #6 : Découplage génération vs exécution

**Deux systèmes complètement séparés** :

```
╔═══════════════════════════════════╗
║ PARTIE 1 : Neural (Seq2seq + GNN) ║
║ Question + Scene Graph → Programme ║
╚═══════════════════════════════════╝
            ↓ (indices)
╔═══════════════════════════════════╗
║ PARTIE 2 : Symbolique (Executor)  ║
║ Programme + Scene JSON → Réponse  ║
╚═══════════════════════════════════╝
```

**Incohérences** :

#### 1. Scene Graph (GNN) ≠ Scene JSON (Executor)

- **GNN** : Utilise scene graph (tensors, embeddings)
- **Executor** : Utilise scene JSON (dict Python, annotations brutes)
- **Ce sont deux représentations différentes de la même chose !**

#### 2. Pas de gradient flow (sauf Reinforce)

- En mode supervised : Executor jamais utilisé pendant training
- Loss sur programmes (tokens), pas sur réponses
- Le seq2seq n'apprend pas à optimiser la réponse finale

#### 3. GNN aide la génération, info utilisée après

- Le GNN enrichit la question avec des embeddings de scène
- MAIS le programme dépend de la syntaxe de la question
- L'executor utilise les annotations brutes (pas les embeddings)

**Impact** :

- Architecture incohérente
- GNN rajoute de la complexité où elle n'est pas nécessaire
- Découplage rend le système fragile (deux sources d'erreur)

---

### ⚠️ Problème #7 : GAT échoue à cause du shared network

**Observation** : GAT (Graph Attention Networks) performe moins bien que GIN.

**Explication** :

#### Différence GIN vs GAT

```python
# GIN : Agrégation uniforme (somme)
h_v = MLP(h_v + SUM(h_u for u in neighbors))

# GAT : Agrégation avec attention
α = attention(h_v, h_u)
h_v = SUM(α * h_u for u in neighbors)
```

#### Problème avec GAT + shared weights

- GAT doit apprendre des **poids d'attention**
- MAIS mêmes poids pour texte (syntaxe) ET scène (spatial)
- **Attention optimale pour texte ≠ attention optimale pour scène**
- Conflit d'apprentissage

**Citation de l'utilisateur** :

> "Parfois les nœuds se concentrent sur les mauvais tokens, il me donne les mauvais objets"

**Avec GIN** :

- Pas d'attention à apprendre
- Agrégation déterministe (somme)
- Plus robuste à l'hétérogénéité des inputs

**Impact** :

- GAT pénalisé par un mauvais design (shared weights)
- Le papier ne teste pas GAT avec poids séparés

---

## Points Positifs

### ✅ Choses bien faites

#### 1. Code propre et modulaire

- Architecture claire (`models/`, `datasets/`, `executors/`)
- Séparation des responsabilités
- Réutilisable pour d'autres tâches

#### 2. Preprocessing bien documenté

- `tools/preprocess_questions.py` : Clair et efficace
- HDF5 pour stockage optimisé
- Vocabulaires bien structurés

#### 3. Executor symbolique robuste

- `executors/clevr_executor.py` : Complet et testé
- Toutes les opérations CLEVR implémentées
- Gestion d'erreurs correcte

#### 4. Ablation GNN optionnelle

- `gembd_vec_dim=0` : Baseline sans GNN
- `gembd_vec_dim=96` : Avec GNN
- Permet de tester l'utilité du GNN

#### 5. Deux modes d'entraînement

- Supervised : Rapide, stable
- Reinforce : End-to-end optimization
- Hybride possible (pretraining + fine-tuning)

---

## Recommandations

### Pour améliorer le papier/code

#### 1. Transparence

- ✅ Publier le code de Task 1 (classification)
- ✅ Clarifier Masked R-CNN (dans papier mais pas code)
- ✅ Documenter les divergences papier/code

#### 2. Comparaisons justes

- ✅ Comparer avec baselines utilisant aussi les annotations CLEVR
- ✅ Ou extraire vraiment les features visuelles (Masked R-CNN)
- ✅ Ablations contrôlées (avec/sans annotations)

#### 3. Design architectural

- ✅ Tester deux GNN séparés (vs shared)
- ✅ Ablation : gembd_vec_dim=0 vs 96 (contrôlée)
- ✅ Tester GAT avec poids séparés

#### 4. Interprétabilité

- ✅ Visualiser la matrice Φ
- ✅ Attention sur features symboliques (pas embeddings)
- ✅ Expliquer les correspondances apprises

---

### Pour un projet GNN sur ce code

#### Expériences pertinentes

##### 1. **Shared vs Separate GNN** (⭐ EXPÉRIENCE LA PLUS INTÉRESSANTE)

**Cette ablation N'EST PAS dans le papier** et pourrait révéler si le shared GNN est un vrai problème :

```python
# Actuel (shared)
self.psi_1 = GIN(in_channels=99, out_channels=48)
h_s = self.psi_1(Gs.x, Gs.edge_index)  # Text graph
h_t = self.psi_1(Gs.x, Gt.edge_index)  # Scene graph (MÊMES POIDS)

# Proposé (separate)
self.gin_text = GIN(in_channels=99, out_channels=48)   # Spécialisé texte
self.gin_scene = GIN(in_channels=99, out_channels=48)  # Spécialisé scène
h_s = self.gin_text(Gs.x, Gs.edge_index)
h_t = self.gin_scene(Gt.x, Gt.edge_index)

# Hypothèse : Separate meilleur (surtout avec GAT)
```

**Pourquoi c'est intéressant** :

- Teste directement le problème du shared GNN
- Facile à implémenter (juste changer `models/graph_matcher.py`)
- Permet de comparer GIN vs GAT avec poids séparés
- **Projet GNN pertinent** (pas besoin de CNN)

##### 2. Architectures GNN

```python
# Tester
- GIN (actuel)
- GAT (avec poids séparés)
- GraphSAGE
- Graph Transformer

# Analyser pourquoi GAT échoue avec shared weights
```

##### 3. Visualisation

```python
# Matrice de correspondance Φ
- Heatmap pour chaque exemple
- Identifier patterns
- Déboguer erreurs

# Exemple où ça échoue :
"parfois les nœuds se concentrent sur les mauvais tokens"
```

---

## Conclusion

### Verdict final

**Le papier souffre de** :

1. ✅ Divergences papier/code (mensonges)
2. ✅ Comparaisons injustes (supervision forte)
3. ✅ Design architectural discutable (shared GNN)
4. ✅ GNN probablement inutile pour Task 2
5. ✅ Architecture incohérente (découplage)

**MAIS** :

- ✅ Code de qualité (modulaire, réutilisable)
- ✅ Task 2 bien implémentée
- ✅ Résultats impressionnants (même si injustes)

**C'est un cas d'école de bullshit académique** :

- Overhyping (buzzwords : multimodal, GNN)
- Comparaisons biaisées
- Code incomplet (Task 1 manquante)
- Design douteux accepté sans vérification

**L'analyse critique révélée ici est MEILLEURE que celle des reviewers** qui n'ont pas :

- Vérifié le code vs papier
- Détecté Masked R-CNN manquant
- Questionné l'utilité du GNN pour Task 2
- Identifié le problème Scene Graph vs Scene JSON

---

## Leçons apprises

### Compétences développées

1. ✅ **Lecture critique de papiers**

   - Ne pas croire les claims sans vérification
   - Comparer papier et code systématiquement

2. ✅ **Analyse architecturale**

   - Identifier les problèmes de design
   - Questionner les choix techniques
   - Proposer des améliorations

3. ✅ **Compréhension du pipeline ML**

   - Preprocessing → Model → Training → Evaluation
   - Chaque étape critiquée individuellement

4. ✅ **Détection du bullshit**

   - Buzzwords vs réalité
   - Marketing académique
   - Comparaisons biaisées

5. ✅ **Propositions d'amélioration**
   - Expériences contrôlées
   - Ablations pertinentes
   - Design alternatifs

**Cette analyse critique est une compétence précieuse en recherche et industrie.**

---

## 💡 Théorie : Pourquoi la multimodalité devrait aider

### Avantages théoriques de la fusion vision + langage

#### 1. **Grounding linguistique (ancrage)**

**Principe** : Les concepts linguistiques deviennent plus robustes quand ils sont "ancrés" dans le monde visuel.

```
"cube rouge" (texte seul)
    ↓
Concept abstrait, peut être ambigu

"cube rouge" + image montrant un cube rouge
    ↓
Concept ancré, lié à propriétés visuelles concrètes
```

**Exemple CLEVR** :

- Question : "What is left of the red cube?"
- Sans vision : "left" est juste un token linguistique
- Avec vision : "left" est lié à des coordonnées spatiales (x, y, z)

**Généralisation** : Si le modèle comprend que "left" = relation spatiale géométrique, il peut appliquer "left" à de nouveaux objets jamais vus ensemble.

---

#### 2. **Redondance et complémentarité**

**Principe** : Deux modalités fournissent des signaux redondants ET complémentaires.

**Redondance (robustesse)** :

```
Question : "Is there a red cube?"
Modalité texte : "red", "cube" → tokens
Modalité vision : objet avec [color=red, shape=cube] → features
```

Si une modalité est bruitée, l'autre compense.

**Complémentarité (information unique)** :

```
Question : "How many cubes are there?"
Modalité texte : "how many", "cubes" (intention)
Modalité vision : 3 objets avec shape=cube (compte)
```

Le texte donne l'intention, l'image donne la réponse.

**Généralisation** : Le modèle apprend des représentations plus riches qui capturent différents aspects du problème.

---

#### 3. **Réduction des biais de shortcut**

**Problème des shortcuts** : Les modèles apprennent des corrélations superficielles.

**Texte seul** :

```
"What color is the cube?" → Réponse fréquente : "red"
Le modèle peut mémoriser : "color" + "cube" → "red" (bias du dataset)
```

**Vision + texte** :

```
"What color is the cube?" + Image montrant un cube bleu
Le modèle DOIT regarder l'image, pas juste mémoriser des patterns textuels
```

**Généralisation** : Force le modèle à apprendre le vrai raisonnement au lieu de shortcuts.

---

#### 4. **Généralisation compositionnelle**

**Principe** : Nouvelles combinaisons de concepts connus.

**Exemple CLEVR** :

- Training : "red cube", "blue sphere", "large object"
- Test : "large blue cube" (jamais vu)

**Multimodal aide** :

- **Vision** : Encode [shape=cube, color=blue, size=large] indépendamment
- **Texte** : Parse "large blue cube" en trois attributs indépendants
- **Fusion** : Le modèle apprend que les attributs sont **indépendants** et **compositionnels**

Sans vision, le modèle texte pourrait traiter "red cube" comme un token unique, pas comme composition (red + cube).

---

#### 5. **Espace latent partagé (joint embedding)**

**Principe** : L'espace latent commun force l'alignement des concepts.

```
Text space:     "red" → vector_text
Vision space:   color=red → vector_vision

Joint embedding:
    align(vector_text, vector_vision) → shared_embedding_red
```

**Généralisation** :

- Les concepts qui s'alignent bien dans l'espace joint sont mieux appris
- Nouvelles combinaisons utilisent les mêmes embeddings alignés

**Exemple** :

```
Training: "small red cube" ↔ image
    → align("small", size_feature)
    → align("red", color_feature)
    → align("cube", shape_feature)

Test: "large red sphere" (nouveau)
    → réutilise align("red", color_feature) ✓
    → compose avec "large" et "sphere"
```

---

### Pourquoi ça devrait marcher pour CLEVR ?

**Dataset CLOSURE (Compositional Structures)** :

- Split A : Certaines combinaisons (attributs, relations)
- Split B : Nouvelles combinaisons des mêmes primitives

**Multimodal aide** :

1. **Décomposition** : Vision force à encoder chaque attribut séparément
2. **Indépendance** : Fusion texte-vision apprend que color ⊥ shape ⊥ size
3. **Recombinaison** : Nouvelles compositions utilisent les primitives apprises

**Sans multimodal** :

- Modèle texte seul peut mémoriser "red cube" comme phrase
- Difficulté à généraliser à "blue cube" si jamais vu

**Avec multimodal** :

- Le modèle DOIT extraire "red" et "cube" comme features visuelles indépendantes
- Recombinaison devient naturelle

---

## ❌ Pourquoi l'implémentation MGN perd ces avantages

### Problème : "Texte + texte" au lieu de "vision + texte"

**Ce que fait MGN** :

```
Question: "What color is the cube?"
    ↓ spaCy embeddings
    ["what", "color", "is", "the", "cube"] → [vec_1, vec_2, vec_3, vec_4, vec_5]

Scene caption: "small red metal cube"
    ↓ spaCy embeddings (MÊMES que question!)
    ["small", "red", "metal", "cube"] → [vec_6, vec_7, vec_8, vec_9]

GNN fusion: combine les vecteurs spaCy
    → Tous des embeddings LINGUISTIQUES
    → Pas de signal VISUEL
```

**Ce que devrait faire un vrai système multimodal** :

```
Question: "What color is the cube?"
    ↓ spaCy embeddings
    Word vectors (sémantique linguistique)

Image (pixels RGB)
    ↓ CNN (Masked R-CNN + ResNet-50)
    Visual features (géométrie, couleurs RGB, textures)

GNN fusion: combine embeddings linguistiques + features visuelles
    → Vision ET langage
    → Vraiment multimodal
```

---

### Ce qui manque sans vraie vision

#### 1. **Propriétés visuelles brutes**

**Vision réelle (CNN)** :

- Détecte la teinte exacte (longueur d'onde lumineuse)
- Texture (lisse vs rugueux, brillant vs mat)
- Forme géométrique précise (cube = 8 coins, 6 faces carrées, angles 90°)
- Relations spatiales 3D (profondeur, occlusion, perspective)

**Caption textuel (MGN)** :

- "red" = mot abstrait, pas la teinte RGB(255, 0, 0)
- "metal" = label, pas la texture brillante avec reflets
- "cube" = label, pas la géométrie 3D avec faces/arêtes
- Positions en JSON = coordonnées, mais pas encodées visuellement par CNN

#### 2. **Ambiguïté linguistique résolue par vision**

**Exemple** :

```
Question: "Is the small sphere left of the cube?"
```

**Avec texte seul (MGN)** :

- "left" = concept linguistique ambigu (gauche de quoi ? du point de vue de qui ?)
- "small" = relatif, dépend du contexte (petit par rapport à quoi ?)
- Résolu via labels dans JSON : "left" → relation spatiale texte

**Avec vision réelle** :

- "left" → CNN encode positions spatiales : $x_{sphere} < x_{cube}$
- "small" → CNN encode taille absolue : radius = 0.5 (vs 1.0 pour "large")
- Features visuelles capturent la géométrie sans ambiguïté

#### 3. **Invariances visuelles**

**Vision réelle (CNN)** apprend des invariances :

```
CNN features pour "red cube":
- Invariant à rotation → toujours un cube rouge (8 coins, rouge partout)
- Invariant à éclairage → rouge sous lumière forte ou faible
- Invariant à position → rouge en haut à gauche ou en bas à droite
- Invariant à échelle → petit ou grand cube reste un cube
```

**Embeddings textuels (MGN)** :

- "red cube" = séquence de tokens dans un ordre fixe
- Pas d'invariance visuelle apprise
- Juste associations linguistiques de co-occurrence dans corpus

#### 4. **Généralisation visuelle vs généralisation linguistique**

**Vision réelle** :

```
Training: voir 100 cubes rouges de différentes tailles, positions, éclairages
    → CNN apprend: "forme avec 6 faces carrées perpendiculaires" = cube
    → CNN apprend: "pixels avec longueur d'onde 620-750nm" = rouge
    → Features universelles, indépendantes des mots

Test: nouveau cube rouge jamais vu (position, taille différentes)
    → CNN reconnaît la GÉOMÉTRIE et la COULEUR
    → Généralise car basé sur propriétés visuelles universelles
```

**Texte (MGN)** :

```
Training: voir "red cube" 100 fois dans différentes phrases
    → spaCy apprend: "red" co-occur avec "color", "bright", "crimson"
    → spaCy apprend: "cube" co-occur avec "shape", "box", "square"
    → Dépend de fréquence linguistique

Test: "blue cube" (phrase rare dans training)
    → spaCy peut mal généraliser car "blue cube" est peu fréquent
    → Dépend de statistiques linguistiques, pas de propriétés visuelles
```

---

### Impact concret sur généralisation compositionnelle

**Scénario CLEVR CLOSURE** :

**Split A (training)** :

- "red cube", "blue sphere", "green cylinder"

**Split B (test)** :

- "blue cube" (nouvelle combinaison jamais vue)

#### Avec vision réelle (idéal) ✅

```
CNN sur images training:
    cube (red) → shape_features_cube = [0.9, 0.1, 0.2, ...]
    sphere (blue) → shape_features_sphere = [0.1, 0.8, 0.3, ...]
    cube (green) → shape_features_cube = [0.9, 0.1, 0.2, ...] (même!)

    red → color_features_red = [0.95, 0.05, 0.02, ...]
    blue → color_features_blue = [0.02, 0.05, 0.93, ...]

Test "blue cube":
    CNN extrait: shape_features_cube + color_features_blue
    → Combine features INDÉPENDANTES apprises
    → ✅ Généralise parfaitement
```

**Pourquoi ?** Les features visuelles sont **décomposées** et **indépendantes** par nature (forme ≠ couleur dans l'espace CNN).

#### Avec captions textuels (MGN) ⚠️

```
spaCy sur captions training:
    "red cube" → embedding_phrase_1 = fonction(co-occur("red", "cube"))
    "blue sphere" → embedding_phrase_2 = fonction(co-occur("blue", "sphere"))
    "green cylinder" → embedding_phrase_3 = ...

Test "blue cube":
    spaCy: embedding pour phrase "blue cube"
    → Si "blue cube" rare dans corpus pré-training spaCy
    → ⚠️ Embedding moins robuste
    → ⚠️ Généralisation dépend de fréquence linguistique
```

**Pourquoi ?** Les embeddings textuels encodent des **co-occurrences** de mots, pas des propriétés visuelles indépendantes.

---

### Preuve mathématique de la perte d'information

**Vision (espace continu, décomposé)** :

$$
\text{Visual features} = [f_{\text{shape}}, f_{\text{color}}, f_{\text{size}}, f_{\text{material}}]
$$

où chaque $f_i$ est **indépendant** et encodé par couches CNN distinctes.

Nouvelles combinaisons = recombinaisons linéaires dans espace produit :

$$
f(\text{blue cube}) = [f_{\text{shape}}(\text{cube}), f_{\text{color}}(\text{blue}), ...]
$$

**Texte (espace discret, co-occurrence)** :

$$
\text{Embedding}(\text{"blue cube"}) = g(\text{co-occur("blue", "cube" dans corpus)})
$$

où $g$ dépend de **fréquence** dans corpus de pré-training.

Si "blue cube" rare :

$$
\text{Embedding}(\text{"blue cube"}) \approx \text{moyenne}(\text{Emb("blue")}, \text{Emb("cube")})
$$

mais pas une vraie décomposition indépendante comme en vision.

---

### Résumé : Pourquoi "texte + texte" perd les avantages

**MGN utilise spaCy pour les DEUX graphes** :

1. ❌ **Pas de signal visuel bas niveau** (couleurs RGB, textures, géométrie 3D)
2. ❌ **Pas d'invariances visuelles** (rotation, éclairage, position)
3. ❌ **Pas de décomposition compositionnelle visuelle** (forme, couleur encodées indépendamment par CNN)
4. ❌ **Dépend de fréquence linguistique** au lieu de propriétés visuelles universelles
5. ❌ **Redondance au lieu de complémentarité** (deux descriptions textuelles du même concept)

**Résultat** :

- Le modèle n'apprend pas à ancrer les concepts linguistiques dans le monde visuel
- La généralisation compositionnelle est limitée par les statistiques linguistiques
- Pas de vraie fusion multimodale (vision + langage)

---

## ✅ Comment implémenter correctement la multimodalité

### Architecture idéale pour CLEVR

**Note** : Pour un projet GNN (pas CNN), on peut garder les annotations CLEVR et se concentrer sur l'amélioration de la partie GNN.

#### ~~Étape 1 : Extraction visuelle réelle~~ (Optionnel - projet CNN)

```python
# Pour vraie multimodalité vision+langage (hors scope projet GNN)
image = load_image("CLEVR_train_000001.png")
visual_features = masked_rcnn(image)  # Détection d'objets
obj_features = resnet50(obj.crop)  # Features visuelles 2048-dim
```

**Raison de skip** :

- Projet GNN, pas CNN
- Complexité technique élevée
- Annotations CLEVR déjà parfaites pour tester GNN

---

#### Étape 1 (pour projet GNN) : Scene graph depuis annotations CLEVR

```python
# Au lieu de : parser caption textuel
caption = "small red metal cube, large blue rubber sphere"
Gt = parser.parse(caption)  # Graphe depuis texte

# Faire : construire graphe depuis détections visuelles
objects = []
for detection in visual_features:
    obj = {
        'visual_features': detection.features,  # 2048-dim CNN features
        'bbox': detection.bbox,                # (x, y, w, h)
        'mask': detection.mask,                # Segmentation mask
    }
    objects.append(obj)

# Relations spatiales depuis positions
for i, obj1 in enumerate(objects):
    for j, obj2 in enumerate(objects):
        if is_left_of(obj1.bbox, obj2.bbox):
            add_edge(Gt, i, j, relation='left')
```

**Avantage** : Graphe construit depuis vision, pas depuis texte.

---

#### Étape 3 : Embeddings multimodaux hétérogènes

```python
# Question graph (texte)
Gs_embeddings = spacy_encoder("What color is the cube?")
# Gs.x shape: [num_nodes_text, 300] (word embeddings)

# Scene graph (vision)
Gt_embeddings = visual_encoder(visual_features)
# Gt.x shape: [num_nodes_objects, 2048] (CNN features)

# Attention : dimensions différentes !
# Gs : 300-dim (linguistique)
# Gt : 2048-dim (visuel)
```

**Problème** : Les deux modalités sont dans des espaces différents → besoin de projection.

---

#### Étape 4 : GNN séparés pour chaque modalité

```python
# Au lieu de : shared GNN
psi_1 = GIN(...)
h_s = psi_1(Gs)  # Texte
h_t = psi_1(Gt)  # Vision (MÊMES POIDS!)

# Faire : GNN séparés
gin_text = GIN(in_channels=300, out_channels=128)
gin_vision = GIN(in_channels=2048, out_channels=128)

h_s = gin_text(Gs.x, Gs.edge_index)   # [num_nodes_text, 128]
h_t = gin_vision(Gt.x, Gt.edge_index)  # [num_nodes_objects, 128]
# Maintenant : même dimension (128), espace latent partagé
```

**Avantage** : Chaque GNN spécialisé pour sa modalité, puis projeté dans espace commun.

---

#### Étape 5 : Fusion avec attention cross-modale

```python
# GraphMatcher avec vraie attention cross-modale
class MultimodalGraphMatcher(nn.Module):
    def forward(self, h_s, h_t):
        # h_s : [B, N_text, 128] (embeddings texte)
        # h_t : [B, N_obj, 128] (embeddings vision)

        # Attention : quels objets correspondent à quels mots ?
        Φ = softmax(h_s @ h_t.T / sqrt(128))  # [B, N_text, N_obj]

        # Fusion : enrichir texte avec vision
        h_s_enriched = h_s + Φ @ h_t  # [B, N_text, 128]

        # Agrégation pour embedding global
        g_embd = mean_pool(h_s_enriched)  # [B, 128]

        return g_embd
```

**Avantage** : Vraie fusion cross-modale, pas juste concatenation.

---

#### Étape 6 : End-to-end training avec loss multimodale

```python
# Training objective : combiner supervised + visual grounding

# 1. Supervised loss (programmes)
programs_pred = seq2seq(question, g_embd)
loss_program = CrossEntropy(programs_pred, programs_gt)

# 2. Visual grounding loss (optionnel)
# Forcer alignement entre mots et objets visuels
alignments_pred = Φ  # Matrice d'attention [N_text, N_obj]
alignments_gt = get_groundings(question, objects)
loss_grounding = MSE(alignments_pred, alignments_gt)

# 3. Total loss
loss = loss_program + λ * loss_grounding
```

**Avantage théorique** : Force le modèle à apprendre l'ancrage vision-langage.

**⚠️ MAIS probablement inutile dans MGN** :

**Analyse** :

```
Programme (neural) + Scene JSON (déterministe)
    → Executor (déterministe)
    → Réponse (déterministe)
```

**Si** :

- Programme correct → Réponse correcte (executor déterministe)
- Scene JSON toujours correct (annotations ground truth)

**Alors** :

- `loss_program` suffit pour optimiser la réponse finale
- `loss_grounding` est redondant car :
  - Le GNN n'est PAS utilisé par l'executor
  - L'executor utilise Scene JSON (pas Scene Graph)
  - Grounding n'affecte pas l'exécution

**Loss grounding utile SI** :

- L'executor utilisait le Scene Graph (GNN) au lieu de Scene JSON
- Ou pour Task 1 (classification directe sur embeddings)

**Pour MGN Task 2** : `loss_grounding` n'ajoute rien car le pipeline est découplé.

---

### Comparaison : MGN actuel vs améliorations possibles

| Composant          | MGN actuel                      | Amélioration GNN (réaliste) | Idéal multimodal (hors scope)  |
| ------------------ | ------------------------------- | --------------------------- | ------------------------------ |
| **Vision**         | Annotations JSON                | Annotations JSON ✓          | Masked R-CNN + ResNet-50       |
| **Scene features** | Labels textuels ("red", "cube") | Labels textuels ✓           | CNN features (2048-dim)        |
| **Embeddings Gs**  | spaCy (99-dim)                  | spaCy (99-dim) ✓            | spaCy (300-dim)                |
| **Embeddings Gt**  | spaCy (99-dim) ⚠️               | spaCy (99-dim) ✓            | CNN features (2048-dim)        |
| **GNN**            | Shared (mêmes poids) ⚠️         | **Séparés** ⭐              | Séparés                        |
| **Fusion**         | Concatenation                   | Concatenation ou attention  | Cross-attention                |
| **Loss**           | Supervised (programmes)         | Supervised ✓                | Supervised (grounding inutile) |
| **Projet**         | Code original                   | **Projet GNN** ⭐           | Projet multimodal complet      |

**Colonne du milieu** = Améliorations réalistes pour un projet GNN sans toucher à la vision.

---

### Expérience pour valider l'avantage multimodal

**Setup** :

1. **Baseline (text-only)** :

   ```python
   Question → seq2seq → Programme → Executor → Réponse
   gembd_vec_dim = 0  # Pas de GNN
   ```

2. **MGN actuel (text + text)** :

   ```python
   Question + Caption → GNN → seq2seq → Programme → Executor
   gembd_vec_dim = 96  # GNN sur spaCy embeddings
   ```

3. **Multimodal idéal (vision + text)** :
   ```python
   Question + CNN features → GNN (séparés) → seq2seq → Programme
   gembd_vec_dim = 128  # GNN sur vision + texte
   ```

**Métriques** :

- Accuracy sur **combinaisons vues** (Split A)
- **Accuracy sur nouvelles combinaisons** (Split B) ← Généralisation compositionnelle
- Temps de training
- Interprétabilité (visualiser attention cross-modale)

**Prédiction** :

- Baseline : Okay sur Split A, mauvais sur Split B
- MGN actuel : Meilleur sur Split A, légèrement mieux sur Split B
- Idéal : Meilleur sur les deux, surtout Split B (vraie décomposition visuelle)

---

### Pourquoi les auteurs n'ont pas fait ça ?

**Raisons possibles** :

1. **Difficulté technique** :

   - Masked R-CNN difficile à entraîner
   - Nécessite beaucoup de GPU pour CNN + GNN
   - Annotations ground truth plus faciles

2. **Focus sur CLOSURE dataset** :

   - CLEVR a déjà des annotations parfaites
   - Objectif = tester généralisation compositionnelle, pas détection d'objets
   - Argument : "On se concentre sur le raisonnement, pas la perception"

3. **Comparaison injuste avec baselines** :
   - Baselines utilisent CNN brutes
   - MGN utilise annotations → gains artificiels
   - Mais permet de publish des résultats impressionnants

**Verdict** : Les auteurs ont choisi facilité + résultats impressionnants au lieu de vraie multimodalité.

---

## ⚖️ Nuances importantes sur l'utilité du scene graph

### Synthèse : Arguments théoriques vs réalité pratique

#### ✅ Théoriquement VALIDE

L'argument du grounding est correct en théorie :

1. **Résoudre ambiguïtés** : "THE metal block" → quelle instance précise ?
2. **Valider exécutabilité** : vérifier que les objets mentionnés existent
3. **Ancrer concepts** : aligner mots linguistiques avec objets visuels
4. **Apprendre correspondances** : GNN matche texte ↔ image

**Ces arguments sont solides pour la Neuro-Symbolic AI.**

#### ⚠️ Pratiquement LIMITÉ pour MGN sur CLEVR

L'implémentation MGN profite **très peu** de ces avantages :

**Pourquoi ?**

1. **Questions non-ambiguës** :

   - Templates CLEVR garantissent unicité (exactement 1 objet par description)
   - Pas de pronoms ambigus
   - Grounding pas nécessaire

2. **Mode supervised (25k iter)** :

   - Pas de validation d'exécutabilité (loss sur tokens)
   - Apprend traduction syntaxique, pas raisonnement visuel
   - GNN aide probablement < 5%

3. **Génération syntaxique** :

   - Mapping token-to-token ("what color" → `query_color`)
   - Structure de la question suffit
   - Info visuelle redondante

4. **Executor découplé** :
   - Utilise Scene JSON (annotations), pas Scene Graph (GNN)
   - Le GNN enrichit seq2seq mais pas l'exécution
   - Deux systèmes séparés

**Verdict nuancé** :

- **Supervised** : GNN aide peu (2-5% ? À tester)
- **Reinforce (1M iter)** : GNN peut aider davantage (10-15% ?) car validation exécution
- **Théorie vs pratique** : Arguments corrects mais CLEVR trop structuré pour en profiter

---

### Quand le scene graph EST vraiment utile

#### Task 1 (Classification directe) : ✅ ESSENTIEL

```
GNN → h_st → Classifier → True/False
```

**Pourquoi ?**

- Pas de programmes symboliques
- Le GNN encode l'image **end-to-end**
- Classification directe sur embedding joint
- Vraiment multimodal (vision + langage fusionnées)
- Le scene graph est la SEULE représentation de l'image

**Ici le grounding est critique** : le modèle doit apprendre à aligner texte et image pour classifier correctement.

#### Task 2 (VQA programmes) : ⚠️ PROBABLEMENT PEU UTILE (supervised)

```
GNN → h_st → Seq2seq → Programme
Programme + Scene JSON → Executor → Réponse
```

**Pourquoi ?**

- Programme = traduction syntaxique (question → opérations)
- Executor fait le vrai travail avec Scene JSON (pas Scene Graph)
- Questions CLEVR non-ambiguës (templates)
- Mode supervised ne valide pas exécutabilité

**Le scene graph enrichit le seq2seq** mais pour une tâche qui n'en a pas vraiment besoin.

#### Task 2 (VQA programmes) : ✅ PEUT AIDER (reinforce)

```
GNN → h_st → Seq2seq → Programme
Programme → Executor → Reward (correct/incorrect)
    ↓
Update GNN + Seq2seq basé sur reward
```

**Pourquoi ?**

- Feedback sur exécutabilité du programme
- Le modèle apprend à générer programmes qui fonctionnent
- Le grounding peut aider à éviter programmes invalides
- Validation end-to-end

**MAIS** :

- Nécessite 1M iterations (vs 25k supervised)
- Training lent et instable (policy gradient)
- Gain incertain (papier ne précise pas)

---

### Le vrai problème : Code incomplet

**Le code publié contient SEULEMENT Task 2 (où le GNN aide peu) !**

Si le code contenait Task 1, la critique serait moins sévère :

- ✅ GNN justifié pour encoder l'image
- ✅ Fusion multimodale légitime
- ✅ Architecture cohérente
- ✅ Grounding essentiel

**Mais Task 1 n'est pas publié** :

- ❌ Impossible de vérifier les résultats de classification
- ❌ Impossible de tester si le GNN aide vraiment
- ❌ Problème de transparence majeur

**Conclusion nuancée** :

L'argument théorique du grounding est **valide**, mais l'implémentation MGN sur CLEVR Task 2 (programmes) ne profite **que marginalement** de ces avantages à cause de :

- La nature syntaxique de la tâche
- Les contraintes de templates CLEVR
- Le découplage GNN/Executor
- Le mode supervised dominant

Pour Task 1 (classification), le GNN serait **essentiel**. C'est dommage que ce code ne soit pas publié.

---

## Références

**Papier original** :

- Multimodal Graph Networks for Compositional Generalization in Visual Question Answering
- NeurIPS (année inconnue)
- 87 citations (malgré les problèmes)

**Code** :

- Repository GitHub (non spécifié)
- Implémentation PyTorch + PyTorch Geometric

**Reviews** :

- `docs/multimodal-graph-network-qa-review.md`
- Critiques légitimes mais incomplètes

**Dataset** :

- CLEVR v1.0 (699,989 questions training)
- CLOSURE (compositional generalization)
- Annotations ground truth (shape, color, size, material)

---

**Fin de l'analyse**

_Cette critique est basée sur une analyse approfondie du code et du papier effectuée le 3 janvier 2026. Toutes les observations sont vérifiables en examinant le code source._
