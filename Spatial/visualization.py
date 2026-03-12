import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
from collections import defaultdict

from mirror_geometry import build_hex_mirror_graph


def make_square_patch(center, size):
    """Kwadratowe lustro wycentrowane na (center)."""
    return patches.Rectangle(
        (center[0] - size / 2, center[1] - size / 2), size, size
    )


# ──────────────────────────────────────────────
# Budowa grafu
# ──────────────────────────────────────────────
row_sizes = [14, 16, 18, 20, 22, 24, 26, 28, 26, 24, 22, 20, 18, 16, 14]
adj, pos, n_total = build_hex_mirror_graph(row_sizes)
sq_size = 0.85

print(f"Łącznie luster: {n_total}")

# ──────────────────────────────────────────────
# Wizualizacja 1: Graf sąsiedztwa + kwadraty
# ──────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(16, 14))

edge_lines = []
drawn_edges = set()
for mid, neighbors in adj.items():
    for nid in neighbors:
        edge_key = tuple(sorted((mid, nid)))
        if edge_key not in drawn_edges:
            drawn_edges.add(edge_key)
            edge_lines.append([pos[mid], pos[nid]])

lc = LineCollection(edge_lines, colors='#cccccc', linewidths=0.5,
                    zorder=1, alpha=0.6)
ax.add_collection(lc)

sq_patches = [make_square_patch(pos[mid], sq_size) for mid in range(n_total)]
n_neighbors = [len(adj[mid]) for mid in range(n_total)]

pc = PatchCollection(sq_patches, cmap='YlOrRd', edgecolors='#333333',
                     linewidths=0.8, zorder=2)
pc.set_array(np.array(n_neighbors))
pc.set_clim(2, 6)
ax.add_collection(pc)

for mid in range(n_total):
    x, y = pos[mid]
    ax.text(x, y, str(mid), ha='center', va='center',
            fontsize=4.5, fontweight='bold', color='#222222', zorder=3)

cbar = plt.colorbar(pc, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Liczba sąsiadów', fontsize=12)

ax.set_xlim(-1, max(row_sizes) + 1)
ax.set_ylim(-1, len(row_sizes) * np.sqrt(3) / 2 + 1)
ax.set_aspect('equal')
ax.set_title(f'MAGIC — kwadratowe lustra w układzie heksagonalnym (N={n_total})\n'
             f'Rzędy: {row_sizes}', fontsize=14, fontweight='bold')
ax.set_xlabel('Kolumna', fontsize=11)
ax.set_ylabel('Rząd', fontsize=11)
ax.grid(False)
ax.set_facecolor('#f8f8f8')
plt.tight_layout()
plt.ylim([15, -2])
plt.show()