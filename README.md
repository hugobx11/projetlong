# RoadEye Guardian 2026

## Description
Ce projet s'inscrit dans le cadre du **Projet Long 2026 - RoadEye Guardian**. Il vise à développer une solution de perception coopérative embarquée pour la prévention proactive des accidents routiers.

Le module actuel implémente une **IA embarquée auto-adaptative** (basée sur YOLOv10n) capable de :
1. Détecter les usagers de la route (Voitures, Bus, Cyclistes, Piétons).
2. Qualifier les entités détectées (Extraction de la couleur dominante).
3. Estimer les distances par stéréovision (2 caméras).
4. Fonctionner avec une latence minimale.

## Architecture du dossier

```text
projetlong/
├── pyproject.toml              → Dépendances et configuration du projet Python
├── README.md                   → Documentation principale du projet
├── .gitignore                  → Fichiers et dossiers ignorés par Git
├── yolov10n.pt                 → Modèle pré-entraîné YOLOv10n (poids du réseau de neurones)
├── src/                        → Code source du module de détection
│   ├── detect_video.py         → Script de détection basique sur vidéo avec YOLOv10
│   ├── detect_vid_v2.py        → Script de détection avancée avec extraction de couleur dominante
│   ├── stereo_vision.py        → Module d'estimation de distance par stéréovision (2 caméras)
│   └── stereo_globaltrack.py   → Variante stéréo avec tracking global spatio-temporel (Kalman + Hongrois)
├── Data/                       → Vidéos de test et données d'entrée (symlink vers Google Drive)
└── runs/                       → Résultats de détection générés (vidéos annotées, logs, etc.)
    └── detect/                 → Dossier contenant les sorties de détection YOLO
```

## Description détaillée des fichiers Python

### `src/detect_video.py`

**Objectif** : Script de détection d'entités basique utilisant YOLOv10 pour analyser une vidéo et générer une vidéo annotée avec un résumé des détections.

**Fonctionnalités principales** :
- **Chargement du modèle** : Utilise YOLOv10n (version nano, optimisée pour la vitesse) depuis un fichier local ou télécharge automatiquement le modèle depuis Ultralytics si absent
- **Traitement vidéo** : Lit une vidéo depuis le dossier `Data/` et applique la détection d'objets frame par frame
- **Catégorisation intelligente** : Regroupe les 80 classes COCO de YOLO en 4 méta-catégories :
  - **Piéton** : `person`
  - **Véhicule** : `car`, `motorcycle`, `train`, `boat`
  - **Bus/Camion** : `bus`, `truck`
  - **Cycliste** : `bicycle`
- **Génération de statistiques** : Compte et affiche le nombre de détections par catégorie dans la console
- **Sauvegarde** : Enregistre la vidéo annotée dans `runs/detect/yolov10_video/` avec les bounding boxes et labels

**Paramètres de détection** :
- Seuil de confiance : `0.05` (très bas pour détecter tous les objets, même peu visibles)
- Format de sortie : Vidéo MP4 annotée avec les détections visuelles

**Utilisation** :
```bash
python src/detect_video.py
```

**Fichiers requis** :
- `Data/WhatsApp_Video_1.mp4` : Vidéo source à analyser
- `yolov10n.pt` : Modèle YOLOv10n (téléchargé automatiquement si absent)

---

### `src/detect_vid_v2.py`

**Objectif** : Version améliorée du script de détection qui ajoute l'extraction de la couleur dominante pour chaque entité détectée, permettant une qualification plus précise des objets.

**Fonctionnalités principales** :
- **Détection YOLOv10** : Utilise YOLOv10n pour détecter les objets dans la vidéo
- **Filtrage par classe** : Ne traite que les classes pertinentes pour la route (piétons, cyclistes, voitures, motos, bus, camions) définies dans `CLASS_MAPPING`
- **Extraction de couleur dominante** : Fonction `get_dominant_color()` qui :
  - Extrait la région d'intérêt (ROI) de chaque bounding box
  - Se concentre sur le centre de l'objet (évite les bordures et ombres)
  - Convertit l'image en espace colorimétrique HSV (plus robuste pour la détection de couleur)
  - Analyse la saturation et la luminosité pour distinguer :
    - **Couleurs achromatiques** : Noir, Blanc, Gris (basé sur la saturation < 40)
    - **Couleurs chromatiques** : Rouge, Orange, Jaune, Vert, Bleu, Violet (basé sur la teinte HSV)
  - Gère les cas limites (objets trop petits, zones vides)

**Mapping des classes** :
- `0` : Piéton
- `1` : Cycliste
- `2` : Voiture
- `3` : Moto
- `5` : Bus
- `7` : Camion

**Affichage visuel** :
- Bounding boxes vertes autour des objets détectés
- Labels combinant le type d'objet et sa couleur dominante (ex: "Voiture | Rouge")
- Fond vert semi-transparent pour améliorer la lisibilité du texte

**Paramètres de détection** :
- Seuil de confiance : `0.4` (plus sélectif que la version basique)
- Traitement frame par frame avec OpenCV pour un contrôle précis de la sortie vidéo

**Utilisation** :
```bash
python src/detect_vid_v2.py
```

**Fichiers requis** :
- `Data/onboard_driving_New_York.mp4` : Vidéo source à analyser
- `yolov10n.pt` : Modèle YOLOv10n

**Sortie** :
- Vidéo annotée sauvegardée dans `runs/detect/res_detect_vid.mp4`

---

### `src/stereo_vision.py`

**Objectif** : Module d'estimation de distance par stéréovision utilisant deux caméras synchronisées pour calculer la profondeur 3D des objets détectés.

**Fonctionnalités principales** :
- **Classe `StereoDepthEstimator`** : Implémente l'estimation de distance stéréoscopique
  - **Calibration des caméras** : Utilise des matrices de projection (P1, P2) pour modéliser la géométrie des deux caméras
  - **Paramètres configurables** :
    - `baseline` : Distance entre les deux caméras (par défaut 0.5 mètres)
    - `focal_length` : Focale en pixels (par défaut 1200.0)
  - **Matrices intrinsèques** : Calcule automatiquement la matrice K (paramètres internes de la caméra) basée sur la résolution supposée (1920x1080)

- **Détection et tracking** :
  - Utilise YOLOv10 pour la détection d'objets sur les deux flux vidéo
  - Intègre ByteTrack pour le suivi temporel des objets (persistance entre les frames)
  - Assure la cohérence des identifiants d'objets entre les frames

- **Triangulation stéréoscopique** :
  - Extrait le point central bas de chaque bounding box (point de contact au sol)
  - Utilise `cv2.triangulatePoints()` pour calculer la position 3D (X, Y, Z) de chaque objet
  - Calcule la distance euclidienne depuis la caméra de référence

- **Matching stéréoscopique** :
  - Algorithme de matching naïf (par ordre) pour associer les objets entre les deux vues
  - **Note** : Pour une production, il faudrait implémenter un matching par similarité visuelle (ReID) ou contrainte épipolaire

- **Affichage en temps réel** :
  - Fenêtre combinée montrant les deux vues côte à côte
  - Bounding boxes colorées (bleu pour caméra 1, rouge pour caméra 2)
  - Overlay textuel affichant : type d'objet, distance en mètres, coordonnées X et Z

**Limitations actuelles** :
- Calibration simulée (nécessite une calibration réelle avec un échiquier)
- Matching simpliste (nécessite amélioration pour scènes complexes)
- Résolution fixe supposée (1920x1080)

**Utilisation** :
```bash
python src/stereo_vision.py
```

**Fichiers requis** :
- `Data/video_left.mp4` : Vidéo de la caméra gauche
- `Data/video_right.mp4` : Vidéo de la caméra droite
- `yolov10n.pt` : Modèle YOLOv10n

**Contrôles** :
- Appuyer sur `q` pour quitter l'application

**Applications** :
- Estimation de distance pour la prévention de collision
- Calcul de vitesse relative des objets
- Cartographie 3D de l'environnement routier

---

### `src/stereo_globaltrack.py`

**Objectif** : Variante avancée de stéréovision qui remplace ByteTrack par un **tracker global spatio-temporel**.  
Ce module fusionne les informations des deux caméras et du temps (plusieurs frames) pour maintenir des identifiants d’objets cohérents, même en cas d’occlusion ou de détections manquantes.

**Principales briques logicielles** :
- **`KalmanTrack`** :
  - Représente une piste (un objet suivi dans le temps).
  - Utilise un **filtre de Kalman 3D + 2D** avec 10 états :
    - Position 3D \((X, Y, Z)\),
    - Position d’image \((c_x, c_y)\),
    - Vitesses correspondantes \((v_X, v_Y, v_Z, v_{cx}, v_{cy})\).
  - L’état est corrigé à chaque nouvelle observation (mesures YOLO + triangulation), et prédit pendant les occlusions pour garder une trajectoire fluide.

- **`GlobalTracker`** :
  - Gère l’ensemble des pistes actives et leurs identifiants.
  - **Association stéréo (entre caméras)** :
    - Utilise l’algorithme de l’assignation Hongroise (`linear_sum_assignment`) sur une matrice de coût.
    - Contraintes intégrées :
      - Même classe d’objet à gauche et à droite.
      - **Contrainte épipolaire simplifiée** : les coordonnées verticales doivent rester proches (lignes épipolaires approximées par des horizontales).
      - Ordre horizontal cohérent (caméra 2 à droite → disparité correcte).
      - Différence de taille limitée entre les bounding boxes gauche/droite.
  - **Estimation 3D** :
    - Utilise `cv2.triangulatePoints` et les matrices de projection \(P_1, P_2\) pour reconstruire les coordonnées \((X, Y, Z)\) à partir du centre bas des boîtes.
  - **Association temporelle (entre frames)** :
    - Tous les `KalmanTrack` prédisent leur prochain état.
    - Une nouvelle matrice de coût compare la prédiction avec les nouvelles observations :
      - Distance en pixels entre centres,
      - Différence de profondeur \(Z\).
    - Les observations non assignées créent de nouvelles pistes, celles non observées incrémentent `lost_frames` et sont supprimées après un seuil.

- **`StereoDepthEstimator` (version globale)** :
  - Charge YOLOv10n et configure une calibration simulée (matrice intrinsèque K, baseline, matrices de projection \(P_1, P_2\)).
  - **`extract_detections`** :
    - Convertit la sortie Ultralytics en dictionnaires plus simples (boîte, classe, confiance) en ne gardant que les classes pertinentes (`CLASS_MAPPING`).
  - **`process_videos`** :
    - Ouvre les deux vidéos (caméra gauche et droite).
    - À chaque frame :
      - Exécute la détection YOLO sur chaque vue (`predict`).
      - Formate les détections, effectue l’association stéréo puis l’association temporelle via `GlobalTracker`.
      - Dessine les boîtes et les labels sur chaque vue :
        - ID global,
        - Classe,
        - Distance euclidienne depuis la caméra,
        - Indication visuelle en cas d’occlusion (épaisseur de trait réduite + label adapté).
    - Concatène les vues gauche/droite côte à côte et enregistre la vidéo de sortie dans `runs/stereo/stereo_globaltrack_result.mp4`.

**Cas d’usage privilégiés** :
- Scènes avec **occlusions fréquentes** (véhicules qui se masquent mutuellement).
- Besoin d’**identifiants stables** pour des modules en aval (fusion de trajectoires, prédiction de trajectoire, coopération entre véhicules).
- Analyse de scénarios CARLA (simulations) nécessitant une cohérence temporelle forte.

---

### Dépendances

Le projet utilise les dépendances suivantes (définies dans `pyproject.toml`) :

- **numpy** (>=2.4.2) : Calculs numériques et manipulation de tableaux.
- **ultralytics** (>=8.4.0) : Framework YOLOv10 pour la détection d'objets (YOLOv10n).
- **opencv-python** (>=4.10.0) : Traitement d'images et de vidéos, lecture/écriture de flux, affichage.
- **lapx** (>=0.5.5) : Bibliothèque d'optimisation utilisée notamment par ByteTrack pour le tracking multi-objet.

### Installation rapide

Depuis la racine du projet :

```bash
# Création d'un environnement virtuel (optionnel mais recommandé)
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate

# Installation des dépendances et du paquet en mode editable
pip install -e .
```

Au premier lancement d'un des scripts, le modèle `yolov10n.pt` sera téléchargé automatiquement si le fichier n'est pas déjà présent à la racine.

### Structure des données

- **Entrée** :
  - Vidéos MP4 dans le dossier `Data/`.
  - Pour la stéréovision : `Data/video_left.mp4` et `Data/video_right.mp4` (caméras synchronisées).
- **Sortie** :
  - Vidéos annotées dans `runs/detect/` (par exemple `yolov10_video/` ou `res_detect_vid.mp4`).
- **Modèle** :
  - Fichier `yolov10n.pt` à la racine (poids du modèle YOLOv10n).

### Notes techniques

- Le projet cible Python **3.11+** (cf. `pyproject.toml`).
- Le dossier `Data/` est un **lien symbolique** vers un répertoire Google Drive afin de gérer facilement des vidéos volumineuses.
- Les répertoires `runs/`, `Data/` et les poids de modèles (`*.pt`) sont exclus du suivi Git via `.gitignore` pour éviter de versionner les données lourdes.
- Les scripts sont pensés pour être intégrés dans un système embarqué ou semi-embarqué, avec une priorité donnée à la **latence faible** et à la **robustesse** des chemins d'accès (utilisation systématique de `pathlib.Path`).
