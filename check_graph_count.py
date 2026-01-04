import torch
import h5py

print("=" * 60)
print("Vérification des fichiers")
print("=" * 60)

# Train
train_graph = torch.load('clevr_preprocessed_100/clevr_train_questions_directed_pairdata.pt', weights_only=False)
print(f"\nTrain graphes: {len(train_graph['data_s_list'])} graphes")

with h5py.File('clevr_preprocessed_100/clevr_train_questions.h5', 'r') as f:
    print(f"Train H5: {len(f['questions'])} questions")

# Val
val_graph = torch.load('clevr_preprocessed_100/clevr_val_questions_directed_pairdata.pt', weights_only=False)
print(f"\nVal graphes: {len(val_graph['data_s_list'])} graphes")

with h5py.File('clevr_preprocessed_100/clevr_val_questions.h5', 'r') as f:
    print(f"Val H5: {len(f['questions'])} questions")

print("\n" + "=" * 60)
print("PROBLÈME: Les nombres ne correspondent pas!")
print("Solution: Limiter max_samples aux nombres de graphes")
print("=" * 60)
