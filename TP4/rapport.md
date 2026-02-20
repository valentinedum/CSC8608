# Rapport TP4

## Exercice 1 : Initialisation du TP et smoke test PyG (Cora)

```bash
TP4/
├── configs/
│   ├── baseline_mlp.yaml
│   ├── gcn.yaml
│   └── sage_sampling.yaml
├── src/
│   ├── smoke_test.py
│   └── utils.py
└── rapport.md
```

Ayant updaté ma version de pytorch, n'est pas compatible avec pyg-lib. Il faut revenir à une version antèrieure comme la 2.8.

```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Nous écrivons le script `TP4/src/smoke_test.py` pour vérifier les installations PyTorch, l'accès GPU et le chargement du dataset Cora.

```bash
=== Environment ===
torch: 2.8.0+cu128
cuda available: True
device: cuda
gpu: NVIDIA H100 NVL MIG 1g.12gb
gpu_total_mem_gb: 10.75

=== Dataset (Cora) ===
Downloading...
Processing...
Done!
num_nodes: 2708
num_edges: 10556
num_node_features: 1433
num_classes: 7
train/val/test: 140 500 1000

OK: smoke test passed.
```