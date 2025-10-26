import torch
import matplotlib.pyplot as plt

from task_vectors import TaskVector  # your class

tv1 = TaskVector.load("sports_advice_task_vector.pt")
tv2 = TaskVector.load("financial_advice_task_vector.pt")
tv3 = TaskVector.load("medical_advice_task_vector.pt")

task_vectors = [tv1, tv2, tv3]
names = ["Sports", "Finance", "Health"]

X = torch.stack([tv.flatten().double() for tv in task_vectors], dim=0) 
print(f"Data type after conversion: {X.dtype}")  

X_centered = X - X.mean(dim=0, keepdim=True)

U, S, V = torch.linalg.svd(X_centered, full_matrices=False)

explained_variance_ratio = (S**2) / (S**2).sum()
print("Explained variance ratio (PC1, PC2, PC3):", explained_variance_ratio[:3])

pc1_proj = X @ V[0]
pc2_proj = X @ V[1]

for name, p1, p2 in zip(names, pc1_proj, pc2_proj):
    print(f"{name} -> PC1: {p1.item():.6f}, PC2: {p2.item():.6f}")

def cosine_similarity(tv_a, tv_b):
    a = tv_a.flatten().double()
    b = tv_b.flatten().double()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

print("\n=== Pairwise Cosine Similarity ===")
for i in range(len(task_vectors)):
    for j in range(i + 1, len(task_vectors)):
        cos_sim = cosine_similarity(task_vectors[i], task_vectors[j])
        print(f"{names[i]} vs {names[j]}: {cos_sim:.6f}")

plt.figure(figsize=(6, 6))
for name, p1, p2 in zip(names, pc1_proj, pc2_proj):
    plt.scatter(p1.item(), p2.item(), label=name, s=120)
    plt.text(p1.item() + 0.05, p2.item() + 0.05, name, fontsize=12)

plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
plt.xlabel(f"PC1 ({explained_variance_ratio[0].item():.2%} variance)")
plt.ylabel(f"PC2 ({explained_variance_ratio[1].item():.2%} variance)")
plt.title("Task Vector PCA (PC1 vs PC2)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()