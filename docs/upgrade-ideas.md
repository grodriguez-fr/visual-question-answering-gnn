C'est parti. Voici la **Proposition Technique Détaillée** pour ton projet. C'est le plan "Syllabus Warrior" qui combine les 3 concepts clés de ton cours (GAT, Link Prediction/VGAE, XAI).

Tu peux copier-coller ces descriptions dans ton rapport et utiliser le code comme base.

---

### 1. L'Encodeur : Graph Attention Network (GAT)

*Référence Cours : Séance 3*

**Le Concept :**
L'architecture originale utilise un GIN (Graph Isomorphism Network), conçu pour distinguer des structures globales. Or, en VQA, la réponse dépend souvent d'un détail local (ex: la couleur d'un petit objet). Le GAT introduit un mécanisme d'attention qui permet à chaque nœud de pondérer l'importance de ses voisins.

**La Modification Technique :**
Au lieu de sortir un simple vecteur, notre GAT va jouer le rôle d'**Encodeur Probabiliste** pour le VGAE. Il doit prédire deux choses pour chaque nœud :

1.  (Mu) : Le sens moyen du nœud.
2.  (Log-Var) : L'incertitude sur ce sens (flou sémantique ou visuel).

**Le Code (Fichier `model.py` - Classe `GNN`) :**

```python
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class ProbabilisticGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=1):
        super().__init__()
        # Couche 1 : Traitement initial et mélange d'info
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        
        # Couches de sortie (Encodeur Variational)
        # Une tête pour la moyenne (le vecteur "sens")
        self.conv_mu = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        # Une tête pour la variance (le vecteur "incertitude")
        self.conv_logvar = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        # 1. Propagation avec Attention
        x = F.elu(self.conv1(x, edge_index))
        
        # 2. Prédiction des paramètres de la gaussienne
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        
        return mu, logvar

```

---

### 2. Le Matcher : Variational Graph Auto-Encoder (VGAE)

*Référence Cours : Séance 6 (Link Prediction)*

**Le Concept :**
Nous reformulons le matching Texte-Image comme un problème de **Link Prediction sur un graphe bipartite**. Au lieu d'un calcul déterministe, nous utilisons un VGAE pour projeter les nœuds dans un espace latent stochastique. Cela permet au modèle de gérer l'ambiguïté (ex: "Objet gris" vs "Objet métallique") en modélisant une distribution de probabilité plutôt qu'un point fixe.

**La Modification Technique :**
On remplace le produit scalaire simple par un échantillonnage (Sampling) suivi d'une prédiction de lien.

**Le Code (Fichier `model.py` - Classe `GraphMatcherVGAE`) :**

```python
class GraphMatcherVGAE(torch.nn.Module):
    def __init__(self, gnn_encoder, hidden_dim):
        super().__init__()
        self.gnn = gnn_encoder # Notre ProbabilisticGAT
        # MLP final pour traiter le résultat du matching
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1) # ou hidden_dim selon le reste du code
        )

    def reparameterize(self, mu, logvar):
        """L'astuce du VGAE (Slide 28)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std  # z = mu + bruit * sigma
        else:
            return mu  # En test, on prend la moyenne (mode déterministe)

    def forward(self, x_txt, edge_txt, x_img, edge_img):
        # 1. Encodage (Sortie: mu, logvar)
        mu_t, logvar_t = self.gnn(x_txt, edge_txt)
        mu_i, logvar_i = self.gnn(x_img, edge_img)

        # 2. Sampling (Espace Latent Z)
        z_txt = self.reparameterize(mu_t, logvar_t)
        z_img = self.reparameterize(mu_i, logvar_i)

        # 3. Probabilistic Link Prediction (Matching)
        # On calcule la matrice d'alignement sur les vecteurs latents Z
        # Matrice [Nb_Mots, Nb_Objets]
        match_matrix = torch.matmul(z_txt, z_img.transpose(-1, -2))
        
        # Normalisation (Optionnelle, aide à la convergence)
        attn_weights = torch.softmax(match_matrix, dim=-1)

        # 4. Projection (On "tire" l'info de l'image vers le texte)
        # Pour chaque mot, on crée un vecteur qui est la somme pondérée des objets
        projected_img = torch.matmul(attn_weights, z_img)
        
        # Fusion (Hadamard product ou Concat)
        multimodal_vec = z_txt * projected_img 
        
        output = self.mlp_out(multimodal_vec)

        # On retourne TOUT pour la Loss et la Viz
        return output, attn_weights, mu_t, logvar_t, mu_i, logvar_i

```

**La Loss Function (ELBO) :**
Dans ta boucle d'entraînement, tu devras ajouter ce terme de perte :

```python
def kl_loss(mu, logvar):
    # Formule Slide 28 : KL(q(z|x) || p(z))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Dans training_loop:
out, attn, mu_t, logvar_t, mu_i, logvar_i = model(...)
loss_classif = criterion(out, target)
loss_kl = kl_loss(mu_t, logvar_t) + kl_loss(mu_i, logvar_i)

total_loss = loss_classif + (0.001 * loss_kl) # On pondère la KL (Beta-VAE)

```

---

### 3. Le XAI : Multimodal Highlighting

*Référence Cours : Séance 8 (Explaining GNNs)*

**Le Concept :**
On ouvre la "Boîte Noire" en visualisant la matrice `attn_weights` générée par le VGAE. Cette matrice nous dit, pour chaque mot, quel objet a été "sélectionné" par le mécanisme de Link Prediction.

**L'Algorithme de Visualisation :**

1. **Récupération :** On prend une paire (Image, Question) du test set. On la passe dans le modèle et on récupère `attn_weights` (Taille ).
2. **Highlight Texte (L'Attention Max) :**
* Pour chaque mot, on regarde son score max d'association avec n'importe quel objet (`attn.max(dim=1)`).
* Si ce score dépasse un seuil (ex: 0.8), on surligne le mot en **Rouge**.
* *Interprétation :* "Ce mot a trouvé un objet correspondant dans l'image."


3. **Highlight Objet (Le "Grounded Object") :**
* On cherche la case  maximale de toute la matrice.
* Disons que c'est (Mot="Sphere", Obj="Objet#3").
* On récupère la **Bounding Box** de l'Objet#3 (coordonnées [x, y, w, h] fournies par le parser).
* On dessine un rectangle **Jaune** sur l'image originale à cet endroit.


4. **Bonus Incertitude (Spécifique VGAE) :**
* On regarde la variance `logvar_i` de l'objet sélectionné.
* Si elle est élevée, on dessine le rectangle en **Pointillés**.
* *Interprétation :* "Je pense que c'est cet objet, mais je ne suis pas sûr (ex: objet flou ou question bizarre)."



**Résultat Visuel (La Slide Démo) :**
Tu montres une image avec la question en dessous.

* Le mot "Shiny" est surligné.
* L'objet brillant dans l'image est encadré.
* Une légende dit : "Match Probability: 98%".

C'est un projet complet, cohérent et visuel. Tu as tout le code logique ici. Tu te sens prêt à l'assembler ?