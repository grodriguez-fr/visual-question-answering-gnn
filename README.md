# visual-question-answering-gnn

## Attribution / Source

Ce dépôt est **basé sur** et **dérive du code** du projet original **MGN** :

- Source (repo original) : https://github.com/raeidsaqur/mgn/

Tout le crédit pour l’implémentation originale revient aux auteurs/mainteneurs du dépôt ci-dessus.

> Remarque : ce dépôt contient des adaptations/expérimentations dans le cadre de ce workspace. Pour la référence officielle, la documentation et la licence, se référer au dépôt source.

## Configuration du Dataset CLEVR

### Téléchargement

Avant d'exécuter le notebook, vous devez télécharger le dataset CLEVR v1.0 :

1. **Lien officiel** : https://cs.stanford.edu/people/jcjohns/clevr/
2. **Télécharger** : `CLEVR_v1.0.zip` (~18 GB)
3. **Extraire** le contenu dans le dossier `data/`

### Structure attendue

```
data/
  CLEVR_v1.0/
    questions/
      CLEVR_train_questions.json
      CLEVR_val_questions.json
    scenes_parsed/
      train_scenes_parsed.json
      val_scenes_parsed.json
```

> **Note** : Le notebook est configuré pour traiter un échantillon limité (`max_sample=100`) pour un développement rapide. Vous n'avez pas besoin du dataset complet pour tester le code.
