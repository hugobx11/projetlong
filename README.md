# RoadEye Guardian 2026

## Description
Ce projet s'inscrit dans le cadre du **Projet Long 2026 - RoadEye Guardian**. Il vise à développer une solution de perception coopérative embarquée pour la prévention proactive des accidents routiers.

Le module actuel implémente une **IA embarquée auto-adaptative** (basée sur YOLOv10n) capable de :
1. Détecter les usagers de la route (Voitures, Bus, Cyclistes, Piétons).
2. Qualifier les entités détectées (Extraction de la couleur dominante).
3. Fonctionner avec une latence minimale.

# Architecture du dossier

```text
projetlong/
├── pyproject.toml          → Dépendances et configuration du projet Python
├── README.md               → Documentation principale du projet
├── src/                    → Code source du module de détection
│   ├── detect_video.py     → Script principal de détection sur flux vidéo / fichier
│   └── detect_vid_v2.py    → Variante / version expérimentale du script de détection
├── Data/                   → Vidéos de test et données d’entrée
└── runs/                   → Résultats de détection générés (vidéos annotées, logs, etc.)
```