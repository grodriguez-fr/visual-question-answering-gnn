# Architecture du Classifier - Alignée avec le papier MGN

## 📚 Référence du papier

**Multimodal Graph Networks for Compositional Generalization in Visual Question Answering**  
_Raeid Saqur, Karthik Narasimhan (NeurIPS 2020)_

## 🎯 Architecture implémentée

### Pour le VQA multi-classe (Section 3.2 du papier)

Le papier décrit clairement l'architecture pour le VQA :

> "For the VQA tasks, the joint representation, $h_{s,t}$ is fed to an attention-based sequence to sequence (seq2seq) model with an encoder-decoder structure. At time step i, the encoder takes a question $q_i$ containing variable length word tokens with added padding and the multimodal $G_s$, $G_t$ representation $h_{s,t}$ as input. The concatenated input $x_i \leftarrow [q_i; h_{s,t}]$ is encoded..."

### Notre implémentation dans `ClassifierModel`

```
                                    CLASSIFIER ARCHITECTURE
                                    ======================

Input:
├─ Questions (x)              → [batch, seq_len]
└─ Graphs (Gs, Gt)            → Tuple of PyG Batch objects

Step 1: Graph Processing
└─ GraphMatcher (GNN)
   ├─ psi_1: GAT/GIN for questions graph
   ├─ psi_2: GAT/GIN for scene graph (if two_stream)
   └─ Output: g_embd [batch, gembd_vec_dim=96]

Step 2: Question Encoding (🔥 NOUVEAU!)
└─ Encoder LSTM
   ├─ Input: [question_embedding; g_embd]  ← Concatenation comme dans le papier!
   ├─ Bidirectional LSTM (2 layers, hidden=256)
   └─ Output: encoder_hidden [batch, hidden*2=512]

Step 3: Classification
└─ Classification Head (MLP)
   ├─ Linear(512 → 256) + ReLU + Dropout
   ├─ Linear(256 → 128) + ReLU + Dropout
   └─ Linear(128 → num_classes) → Logits [batch, 28]

Output: Answer logits → CrossEntropyLoss
```

## 🔄 Différences avec l'ancienne version

### ❌ Version précédente (incorrecte)

```
Graphs (GNN) → g_embd → MLP → answer
```

**Problème** : Ignorait complètement les questions encodées en séquence !

### ✅ Version actuelle (alignée avec le papier)

```
Questions (LSTM) + Graphs (GNN) → Encoder → MLP → answer
```

**Solution** : Utilise les deux modalités comme décrit dans le papier.

## 📊 Composants du modèle

### 1. GraphMatcher (GNN)

- **Type** : GAT ou GIN (configurable)
- **Input** : Graphes de question (Gs) et de scène (Gt)
- **Output** : Embedding multimodal `g_embd` de dimension 96
- **Two-stream** : Peut utiliser deux GNN séparés (comme dans le papier)

### 2. Encoder LSTM

- **Architecture** : Bidirectional LSTM (2 layers)
- **Hidden size** : 256 (donc 512 avec bidirectionnel)
- **Input** : Concatenation de `[question_tokens; g_embd]`
- **Rôle** : Encode la séquence de question en intégrant l'information visuelle

### 3. Classification Head

- **Architecture** : MLP 3 couches
- **Input dim** : 512 (encoder hidden \* 2 si bidirectionnel)
- **Hidden dims** : 256 → 128 → num_classes
- **Dropout** : 0.1 entre chaque couche

## 🔬 Équations du papier implémentées

### Équation (6) : Encoder

```
ei = [ei^F, ei^B]
ei^F, hi^F = LSTM(ψE(xi), hi-1^F)
ei^B, hi^B = LSTM(ψE(xi), hi+1^B)
```

où `xi ← [qi; hs,t]` (concatenation question + graph embedding)

**Notre implémentation** :

```python
# Dans Encoder.__init__
rnn_input_dim = self.word_vec_dim + self.gembd_vec_dim

# Dans Encoder.forward
g_embedding = g_embd.unsqueeze(1).repeat(1, x.size(1), 1)
word_embedding = self.embedding(x)
embedded = torch.cat((word_embedding, g_embedding), dim=2)
```

### Classification finale

Au lieu de générer un programme (comme dans le mode seq2seq), on utilise directement le hidden state de l'encoder pour la classification :

```python
# Dans ClassifierModel.forward
encoder_outputs, encoder_hidden = self.encoder(self.x, g_embd)
h = encoder_hidden[0]  # Take hidden state
logits = self.classifier(h)
```

## 📈 Avantages de cette architecture

1. **✅ Utilise les deux modalités** : Questions (séquence) + Graphes (structure)
2. **✅ Suit le papier original** : Architecture décrite en Section 3.2
3. **✅ Meilleure capacité de généralisation** : L'encoder LSTM capture la sémantique séquentielle
4. **✅ Fine-grained matching** : Le GNN établit des correspondances fines entre concepts visuels et linguistiques

## 🚀 Utilisation

```python
from models.classifier_model import ClassifierModel

# Créer le modèle
model = ClassifierModel(opt)

# Forward pass
model.set_input(x=questions, g_data=(batch_s, batch_t), answers=answer_labels)
loss = model.supervised_forward()  # Calcule CrossEntropyLoss
model.supervised_backward()         # Backprop

# Prédiction
predictions = model.predict()       # Indices de classes
answer_strings = model.predict_answers()  # Strings de réponses
```

## 📝 Paramètres importants

```python
opt.gembd_vec_dim = 96              # Dimension des embeddings de graphe
opt.decoder_hidden_size = 256       # Hidden size de l'encoder LSTM
opt.encoder_num_layers = 2          # Nombre de couches LSTM
opt.bidirectional_encoder = 1       # Encoder bidirectionnel
opt.classifier_hidden_dim = 256     # Hidden dim du classification head
opt.classifier_dropout = 0.1        # Dropout du classifier
opt.two_stream_gnn = 1             # Deux GNN séparés pour Gs et Gt
```

## 🔍 Différences avec le mode seq2seq

| Aspect           | Seq2seq (Executor)            | Classifier (End-to-End)  |
| ---------------- | ----------------------------- | ------------------------ |
| **Architecture** | Encoder → Decoder → Programme | Encoder → MLP → Réponse  |
| **Output**       | Séquence de tokens programme  | Logits de classification |
| **Loss**         | CrossEntropy sur tokens       | CrossEntropy sur answers |
| **Exécution**    | Symbolic executor             | Direct (pas d'executor)  |
| **Questions**    | Encodées par LSTM ✅          | Encodées par LSTM ✅     |
| **Graphes**      | Utilisés par GNN ✅           | Utilisés par GNN ✅      |

## 📚 Références

- **Paper** : [Multimodal Graph Networks (NeurIPS 2020)](https://arxiv.org/abs/2010.02663)
- **Section 3.2** : Using the multimodal representation in downstream tasks
- **Équation (6)** : Encoder architecture avec concatenation `[qi; hs,t]`
- **Table 1** : Résultats sur Caption Truth Prediction (98.2% accuracy)
