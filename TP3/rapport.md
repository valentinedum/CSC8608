# Rapport TP3

## Exercice 1 : Initialisation du TP3 et vérification de l’environnement

Voici le sanity check :
![sanity_check](./img/Capture%20d’écran%202026-02-19%20101559.png)

## Exercice 2 : Constituer un mini-jeu de données : enregistrement d’un “appel” (anglais) + vérification audio

Nous enregistrons un échantillons d'un appel téléphonique dans `TP3/data/call_01.wav`. Puis nous vérifions ses métadonnées.

**Métadonnées audio :**
- **Durée :** 39.91 secondes (~40s)
- **Sample rate :** 48000 Hz
- **Canaux :** 1 (mono)
- **Codec :** PCM 16-bit signed LE
- **Bitrate :** 768 kb/s

Le fichier est raisonnable en temps, mono mais est en 48kHz , nous le convertissons en 16kHz.

```bash
ffmpeg -i TP3/data/call_01.wav -ar 16000 -ac 1 TP3/data/call_01_16k.wav
mv TP3/data/call_01_16k.wav TP3/data/call_01.wav
```

Nous créons ensuite un script `TP3/inspect_audio.py` pour afficher la forme du tenseur, le sample rate, la durée, et quelques statistiques simples (RMS, taux de clipping).

![inspect_audio](./img/Capture%20d’écran%202026-02-19%20105550.png)

Il n'y a pas clipping, donc l'audio ne continet pas de saturation. Et le RMS est de **rms: 0.0606**. On voit bien qu'on est passé sur du **16kHz** de sample rate ce qui représente environ **600 000** échantillons.

## Exercice 3 : VAD (Voice Activity Detection) : segmenter la parole et mesurer speech/silence

Nous créons, à présent le script `TP3/vad_segment.py` qui utilise un vad prêt à l'emploi pour produire une liste de segments et calculer des statistiques dessus.

Nous avons besoin pour cela d'installer silero_vad :

```bash
pip install silero_vad
```

![vad_exe](./img/Capture%20d’écran%202026-02-19%20110703.png)

**Extrait du JSON (5 premiers segments) :**
```json
{
  "audio_path": "TP3/data/call_01.wav",
  "sample_rate": 16000,
  "duration_s": 39.9146875,
  "min_segment_s": 0.3,
  "segments": [
    {
      "start_s": 0.226,
      "end_s": 0.766
    },
    {
      "start_s": 0.962,
      "end_s": 2.974
    },
    {
      "start_s": 3.426,
      "end_s": 5.758
    },
    {
      "start_s": 6.434,
      "end_s": 9.182
    },
    {
      "start_s": 9.57,
      "end_s": 12.542
    },
    ]
}
```

**Analyse :** Le ratio de parole est **75.8%**, ce qui est cohérent pour un discours naturel avec des pauses courtes (respiration, ponctuation). Les 18 segments détectés sont cohérents avec les 9 phrases dites.

Nous augmentons le filtrage `min_dur_s` pour le rendre plus strict, le faisant passer de **0.3** à **0.6**. Nous avons maintenant **15** segments au lieu de **18** et un speech ratio de **72.1%** contre **75.8%**.