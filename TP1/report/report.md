# Rapport TP1

## Prérequis

## Dépôt et architecture du code

- Lien vers dépot : [Github](https://github.com/valentinedum/CSC8608)
- Lieu d'exécution du TP : **SLURM**
- Architecture du TP :
![architecture](./images/Capture%20d’écran%202026-01-23%20110052.png)

## Setup Environnement

Nom environnement conda : `deeplearning`
Versions :
```bash
torch 2.5.1
cuda_available True
device_count 1
```

Nous installons les `requirements.txt`. SAM est bien installé :
![setup](./images/Capture%20d’écran%202026-01-23%20111926.png)

## Forwarding du port Streamlit

- **Port choisi** : 8542
- **Infrastructure** : Utilisation d'un tunnel SSH avec rebond (Jump Host) pour atteindre le nœud GPU `arcadia-slurm-node-2` via `tsp-client`.
- **Commande de tunnel (exécutée sur PC local)** : 
  `ssh -L 8542:arcadia-slurm-node-2:8542 tsp-client`
- **UI accessible via SSH tunnel** : Oui.
