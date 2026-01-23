# Rapport TP1

## Exercice 1 : Initialisation du dépôt, réservation GPU, et lancement de la UI via SSH

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

## Exercice 2 : Constituer un mini-dataset (jusqu’à 20 images)

On va récuperer des images sur le net pour ces cas intéressants:
- 3 Simples : Une tasse (`simple_mug.jpg`), un smartphone (`simple_smartphone.jpg`), un fruit (`simple_fruit.png`)
![mug](../data/images/simple_mug.jpg)
- 3 Chargées : Un bureau en désordre (`charge_bureau.jpg`), une rue (`charge_rue.jpg`), un rayon de supermarché (`charge_supermarche.jpg`).
![rue](../data/images/charge_rue.jpg)
- 2 Difficiles : Des lunettes (transparence) avec `complexe_lunettes.jpg`, un vélo (rayons fins) avec `complexe_velo.jpg`, ou des cheveux avec `complexe_cheveux.jpg`.
![lunettes](../data/images/complexe_lunettes.jpg)

Puis on va les completer avec des photos aléatoires depuis l'API **picsum** avec le script `TP1/src/generate_dataset.py`

## Exercice 3 : Charger SAM (GPU) et préparer une inférence “bounding box → masque”

### Téléchargement de SAM

![sam](./images/Capture%20d’écran%202026-01-23%20131348.png)

### Test Sam

Après configuration de sam pour prédire les masques, on le teste : 
![sam_test](./images/Capture%20d’écran%202026-01-23%20131826.png)

**Constat suite au test rapide** :
- Réussite technique : Le chargement sur GPU est effectif (pas d'erreur CUDA) et le modèle **vit_h** répond en un temps raisonnable malgré sa taille.
- Comportement de SAM : Le score de **0.86** montre que même avec une bounding box "arbitraire", le modèle parvient à extraire une structure cohérente.

## Exercice 4 : Mesures et visualisation : overlay + métriques (aire, bbox, périmètre)

Nous maintenant produire des sorties “exploitables” : une visualisation claire en créant les scripts `geom_utils.py` et `viz_utils.py`.

Nous pouvons tester que tout fonctionne avec ce tableau des résultats: 

|Image         |Score SAM|Aire (pixels)|Périmètre (pixels)|
|--------------|---------|-------------|------------------|
|random_8.jpg  |0.8622   |17 575       |1152.06           |
|random_2.jpg  |0.8419   |2 958        |654.88            |
|simple_mug.jpg|0.8447   |52 018       |2732.63           |

Voici l'exemple de random_8.jpg : 
![test_box](../outputs/overlays/overlay_random_8.png)

On voit que SAM a reussi à delimiter le gris très sombre du ciel des nuages plus claires.

**Utilité de l'overlay pour le "debug"**
L'overlay (le masque rouge) nous permet de voir ce que l'IA a compris. Sans ça, on aurait juste un score et une aire, mais on ne saurait pas si SAM a bien entouré l'objet ou s'il a fait n'importe quoi.

Par exemple, sur mes tests, ça permet de 'débugger' tout de suite : si je vois que le rouge déborde sur le décor, je sais que ma boîte (le cadre vert) est trop large ou que l'image est trop compliquée pour lui. C'est aussi là qu'on voit si SAM est 'fainéant' : parfois il ne colorie qu'un petit bout de l'objet au lieu de tout prendre. L'image nous montre tout de suite si le modèle a réussi son coup ou s'il s'est perdu dans les détails.


## Exercice 5 : Mini-UI Streamlit : sélection d’image, saisie de bbox, segmentation, affichage et sauvegarde

Nous allons maintenant assembler les briques précédentes dans une mini-UI Streamlit. Pour cela, nous modifions le fichier `TP1/src/app.py`

**Résultats**

|Image         |Cas de test|Score |Aire (px)|Temps (ms)|
|--------------|-----------|------|---------|----------|
|Mug           |Simple (Objet isolé)|0.996 |193 940  |1631.3    |
|Souris        |Précis (BBox serrée)|0.986 |2 630    |1643.2    |
|Souris        |Ambigu (BBox large)|0.952 |10 060   |1634.8    |


### Cas complexe Souris dans environnement chargé

![souris_bien](./images/f4c0f670b59b84e9221baa2eb1d39b8371e5bc2c3be54073844dfd29.jpg)

![souris_mal](./images/f62216ed34ec8a2c5dfab4ffb93d22952c85508fec54811acfe2150d.jpg)

>il faut que la souris soit encadrée précisément pour que l bon objet soit detecter.

### Cas simple Mug

![mug_bien](./images/5d561f024cf823d4840e8214361d7724b9c98a54e3ed7f98d08c981a.jpg)

>Une segmentation quasi parfaite avec un score maximal.

### Impact BBox

L'ajustement de la BBox est déterminant pour la précision de SAM. Une boîte trop large crée de l'ambiguïté : dans mon test sur la souris, le modèle a inclus le tapis de souris, multipliant l'aire par quatre et faisant baisser le score de confiance. À l'inverse, une boîte serrée force SAM à isoler l'objet exact, produisant un masque net et un score optimal (0.98+). L'overlay est donc essentiel pour vérifier que le "prompt" (la boîte) ne capture pas d'éléments parasites du décor.

## Exercice 6 : Affiner la sélection de l'objet : points FG/BG + choix du masque (multimask)

Sans les points d'aide de FG et BG, SAM pense que le mask d'index 2 est le meilleur. Or la personne sur la droite n'est pas le meme objet que la plante.


```json
{"scores":[0.9154070019721985,0.9289053678512573,0.9351819753646851],"time_ms":1642.7719593048096}

{"mask_idx":2,"score":0.9351819753646851,"area_px":6471,"mask_bbox":[282,361,386,477],"perimeter":594.6173115968704}
```

![box_rue_wrong](./images/3a3c52a1c58bef613357e52d6117b372ce166f082d6e459cbc3a9579.jpg)

```json
{"scores":[0.9289184212684631,0.9520488977432251,0.9325963258743286],"time_ms":1646.0275650024414}

{"mask_idx":1,"score":0.9520488977432251,"area_px":5234,"mask_bbox":[282,371,355,477],"perimeter":337.5218583345413}
```