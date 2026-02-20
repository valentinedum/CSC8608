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

## Exercice 2 : Baseline tabulaire : MLP (features seules) + entraînement et métriques

Nous créons les scripts `TP4/src/data.py`, `TP4/src/utils.py`, `TP4/src/models.py` et `TP4/src/train.py`

**Protocole d'évaluation** : Sur Cora, on ne divise pas le dataset en fichiers séparés (pour éviter le data leakage) car on doit garder la structure du graphe intacte. On utilise des masques pour filtrer les nœuds : le **train_mask** sert à l'optimisation des poids, le **val_mask** permet de prévenir l'overfitting et de régler nos hyperparamètres, et le **test_mask** sert de juge final pour mesurer la généralisation réelle sans biais. Calculer les métriques sur chaque masque à chaque epoch nous permet de valider la convergence du modèle en temps réel tout en s'assurant qu'il ne commence pas à overfitter.


### Configuration et résultats MLP

Configuration utilisée : `TP4/configs/baseline_mlp.yaml`
```yaml
seed: 42
device: "cuda"
epochs: 100
lr: 0.001
weight_decay: 5e-4
mlp:
  hidden_dim: 128
  dropout: 0.5
```

Entraînement et métriques finales :
```bash
epoch=001 loss=1.9471 train_acc=0.2643 val_acc=0.1540 test_acc=0.1510 train_f1=0.2258 val_f1=0.0809 test_f1=0.0805 epoch_time_s=0.8972
epoch=020 loss=1.4013 train_acc=0.9929 val_acc=0.5500 test_acc=0.5580 train_f1=0.9929 val_f1=0.5304 test_f1=0.5416 epoch_time_s=0.0018
epoch=040 loss=0.5711 train_acc=1.0000 val_acc=0.5820 test_acc=0.5750 train_f1=1.0000 val_f1=0.5672 test_f1=0.5596 epoch_time_s=0.0017
...
epoch=100 loss=0.0682 train_acc=1.0000 val_acc=0.5820 test_acc=0.5670 train_f1=1.0000 val_f1=0.5716 test_f1=0.5549 epoch_time_s=0.0041
total_train_time_s=1.0773
```

**Résultats finaux (test set)** :
- **Accuracy** : 0.5670
- **Macro-F1** : 0.5549
- **Total training time** : 1.0773 s

**Observations** : Le MLP atteint l'overfitting au-delà de l'epoch 20 (train_acc=1.0 dès epoch 40, val_acc plafonné ~0.58), ce qui suggère que les features seules ne capturent pas suffisamment la structure du graphe. Les performances test/val restent similaires (0.567/0.582), validant la procédure d'évaluation.

## Exercice 3 : Baseline GNN : GCN (full-batch) + comparaison perf/temps

Cette fois-ci nous mettons à jour `TP4/src/data.py` pour exposer aussi edge_index, ajoutons la classe GCN dans `models.py`, et refactorisons `train.py` avec un flag `--model {mlp,gcn}` pour supporter les deux modèles avec la même configuration (seed 42, lr 0.001, hidden_dim 128, dropout 0.5).

Entraînement et métriques finales :
```bash
epoch=001 loss=1.9443 train_acc=0.4071 val_acc=0.2080 test_acc=0.2480 train_f1=0.3998 val_f1=0.1728 test_f1=0.2149 epoch_time_s=0.5035
epoch=020 loss=1.0822 train_acc=0.9857 val_acc=0.7700 test_acc=0.7870 train_f1=0.9859 val_f1=0.7536 test_f1=0.7793 epoch_time_s=0.0028
epoch=040 loss=0.3861 train_acc=0.9857 val_acc=0.7820 test_acc=0.8010 train_f1=0.9859 val_f1=0.7651 test_f1=0.7965 epoch_time_s=0.0028
...
epoch=100 loss=0.0574 train_acc=1.0000 val_acc=0.7840 test_acc=0.7950 train_f1=1.0000 val_f1=0.7713 test_f1=0.7921 epoch_time_s=0.0029
total_train_time_s=0.7891
```

**Résultats finaux (test set)** :
- **Accuracy** : 0.7950
- **Macro-F1** : 0.7921
- **Total training time** : 0.7891 s

### Comparaison MLP vs GCN

| Métrique | MLP | GCN | Gain |
|----------|-----|-----|------|
| Test Accuracy | 0.5670 | 0.7950 | +40% |
| Test Macro-F1 | 0.5549 | 0.7921 | +43% |
| Training time (s) | 1.0773 | 0.7891 | -27% (plus rapide) |

### Analyse : Pourquoi GCN surpasse MLP

Le gain de **+40%** s'explique par l'homophilie du graphe. Sur Cora, les articles ont tendance à citer des papiers de la même catégorie. Le MLP n'est pas bon parce qu'il traite chaque nœud comme s'il était isolé, en se basant uniquement sur son "bag-of-words". Le GCN utilise le message passing pour intégrer les features des voisins dans le calcul.

Ce mécanisme lui permet de compenser des features locales parfois bruitées ou insuffisantes par le contexte global du voisinage. Alors que le MLP sature vite et fait de l'overfitting (il apprend par cœur les mots-clés du train sans comprendre les liens), le GCN capture le signal du graphe. Donc même si les deux modèles montent à 1.0 d'accuracy sur le train, le GCN généralise  mieux sur le test set car ses représentations sont enrichies par le lien entre les citations, là où le MLP reste "aveugle" aux relations.

## Exercice 4 : Modèle principal : GraphSAGE + neighbor sampling (mini-batch)

Configuration de sampling : `TP4/configs/sage_sampling.yaml`
```yaml
seed: 42
device: "cuda"
epochs: 100
lr: 0.001
weight_decay: 5e-4

sage:
  hidden_dim: 128
  dropout: 0.5

sampling:
  batch_size: 128
  num_neighbors_l1: 32
  num_neighbors_l2: 16
```

Entraînement et métriques finales :
```bash
epoch=001 loss=1.9556 train_acc=0.5429 val_acc=0.1760 test_acc=0.2010 train_f1=0.5086 val_f1=0.1240 test_f1=0.1907 epoch_time_s=0.2860
epoch=010 loss=1.1665 train_acc=1.0000 val_acc=0.7140 test_acc=0.7150 train_f1=1.0000 val_f1=0.7190 test_f1=0.7157 epoch_time_s=0.0037
epoch=020 loss=0.3611 train_acc=1.0000 val_acc=0.7500 test_acc=0.7720 train_f1=1.0000 val_f1=0.7553 test_f1=0.7715 epoch_time_s=0.0037
...
epoch=050 loss=0.0328 train_acc=1.0000 val_acc=0.7780 test_acc=0.7970 train_f1=1.0000 val_f1=0.7721 test_f1=0.7920 epoch_time_s=0.0037
epoch=100 loss=0.0096 train_acc=1.0000 val_acc=0.7740 test_acc=0.8060 train_f1=1.0000 val_f1=0.7647 test_f1=0.7996 epoch_time_s=0.0037
total_train_time_s=0.7006
```

**Résultats finaux (test set)** :
- **Accuracy** : 0.8060
- **Macro-F1** : 0.7996
- **Total training time** : 0.7006 s

### Comparaison MLP vs GCN vs GraphSAGE

| Métrique | MLP | GCN | GraphSAGE | Gain SAGE/MLP |
|----------|-----|-----|-----------|--------------|
| Test Accuracy | 0.5670 | 0.7950 | 0.8060 | +42% |
| Test Macro-F1 | 0.5549 | 0.7921 | 0.7996 | +44% |
| Training time (s) | 1.0773 | 0.7891 | 0.7006 | -35% (2x plus rapide) |

**Conclusion**
Le neighbor sampling permet d'entraîner le modèle sans charger tout le graphe d'un coup. Au lieu de prendre tous les voisins (ce qui ferait ramer le GPU sur les gros nœuds très connectés), on fixe un fanout : on prend au hasard 32 voisins au premier étage et 16 au second. Ça stabilise la charge de calcul et ça explique pourquoi l'entraînement est plus rapide (0.70s, soit 2x plus vite que le MLP).

C’est un compromis "vitesse vs précision" : en ne prenant qu'un échantillon, on introduit de la variance dans le gradient car le modèle ne voit qu'une partie du voisinage à chaque batch. Il y a aussi un petit coût CPU pour gérer le tirage aléatoire des voisins. Mais sur Cora, ça marche : l'accuracy monte un peu (**80.6%**). Ce "bruit" introduit par le sampling semble même aider à mieux généraliser que le GCN classique, prouvant que GraphSAGE est meilleur pour des graphes géants et très performant pour capter le signal du voisinage sans s'éparpiller.

## Exercice 5 : Benchmarks ingénieur : temps d’entraînement et latence d’inférence (CPU/GPU)

Nous modifions le train pour laisser un checkpoint des 3 modèles dans `TP4/runs/`
Puis, nous exécutons le script de benchmarks sur les 3 modèles en utilisant les checkpoints.

Résultats des benchmarks (100 runs par modèle, GPU H100) :
```
model: mlp
device: cuda
avg_forward_ms: 0.0943
num_nodes: 2708
ms_per_node_approx: 3.482e-05

model: gcn
device: cuda
avg_forward_ms: 1.3675
num_nodes: 2708
ms_per_node_approx: 0.00050499

model: sage
device: cuda
avg_forward_ms: 0.4119
num_nodes: 2708
ms_per_node_approx: 0.0001521
```

### Tableau synthétique : Performance vs Latence

| Modèle | Test Accuracy | Test F1 | Train time (s) | Inference (ms) |
|--------|---------------|---------------|----------------|----------------|
| MLP    | 0.5670        | 0.5549        | 1.0773         | 0.0943         |
| GCN    | 0.7950        | 0.7921        | 0.7891         | 1.3675         |
| GraphSAGE | 0.8060     | 0.7996        | 0.7006         | 0.4119         |

### Importance du warmup et de la synchronisation CUDA

On fait un warmup pour préparer la carte GPU. Sans ça, le premier calcul est toujours très lent parce que le système doit tout mettre en place (chargement bibliotheques, etc), ce qui fausserait les moyennes. Le warmup permet donc des mesures stables et réalistes.
Pour la synchronisation, c'est parce que le GPU travaille de manière asynchrone. Quand on lui donne un calcul, le code python n'attend pas la fin et continue d'avancer tout seul. Si on n'utilise pas le `torch.cuda.synchronize()`, on va chronométrer le temps que met le processeur à donner l'ordre, et pas le temps effectif d'execution de la tâche.

## Exercice 6 : Synthèse finale : comparaison, compromis, et recommandations ingénieur

| Modèle      | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|------------|----------|---------------|--------------------|----------------|----------------|
| MLP        | 0.5670   | 0.5549        | 1.0773             | ~1.60          | 0.0943         |
| GCN        | 0.7950   | 0.7921        | 0.7891             | ~1.17          | 1.3675         |
| GraphSAGE  | 0.8060   | 0.7996        | 0.7006             | 1.1755         | 0.4119         |

### Recommendation ingénieur

D'après mes mesures, le choix du modèle dépend du compromis entre précision et ressources. Le MLP est à garder uniquement pour sa rapidité. C'est le plus rapide en inférence (**0,09 ms**), mais sa qualité est trop faible (**56% d'accuracy**) car il ignore les liens du graphe.
Pour de la performance pure, je recommande GraphSAGE, qui arrive en tête avec **80,6% d'accuracy**. En plus d'être le plus précis, il est environ 3 fois plus rapide que le GCN en latence (**0,41 ms** contre **1,37 ms**) grâce au neighbor sampling qui limite le nombre de voisins à traiter. Le GCN reste une alternative robuste avec **79,5% d'accuracy**, mais sa lenteur en traitant tous les voisins le rend moins intéressant pour une mise en production à grande échelle.
En résumé, pour un système réel alliant efficacité et précision, GraphSAGE est le meilleur choix d'ingénierie d'après mes mesures. Ce serait à confirmer en faisant d'autres tests avec des configurations différentes.

### Risque de protocole

Un risque majeur qui pourrait fausser la comparaison est le Data Leakage. Si on utilise par erreur des arêtes du "test set" pendant la phase d'antrainement soit de message passing, le modèle triche en voyant des connexions qu'il n'est pas censé voir. Ce qui gonfle artifiellement l'accuracy et la fausse.
Dans un vrai projet, j'éviterais cela en isolant strictement le graphe vu par le modèle pendant l'entrainement. Un autre risque pourrait être le manque de reproducibilité lié au sampling. Pour garantir des mesures comparables entre modèles, il est nécessaire de fixer des seeds pour bloquer le côté aléatoire.

### Check rapide du dépôt

Les checkpoints ont été mis dans le gitignore et le dataset est présent dans le cache.
