import json

vocab_100 = json.load(open('clevr_preprocessed_100/clevr_vocab.json'))
print(f"Vocab 100 samples: {len(vocab_100['question_token_to_idx'])} mots")

# Vérifier s'il existe un vocab complet
try:
    vocab_full = json.load(open('clevr_preprocessed/clevr_vocab.json'))
    print(f"Vocab complet: {len(vocab_full['question_token_to_idx'])} mots")
except:
    print("Pas de vocab complet trouvé")
