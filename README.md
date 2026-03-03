## RoadEye Guardian 2026

### Description

Ce projet s'inscrit dans le cadre du **Projet Long 2026 - RoadEye Guardian**.  
Il vise à développer une solution de **perception coopérative embarquée** pour la prévention proactive des accidents routiers.

Le module actuel implémente une **chaîne de stéréovision + tracking global** basée sur YOLO (modèle léger) et un filtre de Kalman 3D, capable de :

1. **Détecter** les usagers de la route pertinents (Piétons, Cyclistes, Voitures, Motos, Bus, Camions).
2. **Estimer la distance 3D** des objets par triangulation stéréo (2 caméras).
3. **Suivre les objets dans le temps** avec un tracker global spatio‑temporel robuste aux occlusions.
4. **Visualiser la scène en vue de dessus** (plan \(X\)-\(Z\)) pour analyser les trajectoires.

---

### Architecture du dossier

```text
projetlong/
├── pyproject.toml            → Dépendances et configuration du projet Python
├── README.md                 → Documentation principale du projet
├── .env                      → Variables d'environnement pour la stéréovision / tracking / vue top-down
├── .gitignore                → Fichiers et dossiers ignorés par Git
├── src/                      → Code source du module de stéréovision
│   ├── stereo_globaltrack.py → Script principal : stéréovision + tracking global + vue top-down
│   ├── tracker.py            → Implémentation du tracker global (Kalman + Hongrois)
│   ├── topdown_view.py       → Visualisation en vue de dessus (Matplotlib, optionnel)
│   ├── stereo_calibration.py → Gestion d'une calibration stéréo (chargement/rectification)
│   └── calibrate_stereo.py   → Outil CLI pour estimer une calibration stéréo (.npz) à partir d'un damier
├── Data/                     → Vidéos de test et données d'entrée
│   └── Simulation_x/
│       ├── gauche.mp4        → Vidéo caméra gauche
│       └── droite.mp4        → Vidéo caméra droite
└── runs/                     → Résultats générés (vidéos annotées, etc.)
    └── stereo/
        └── Simulation_x/
            └── Simulation_x.mp4
```

> **Remarque** : dans la configuration par défaut, le nom de la simulation est piloté par la variable d'environnement `VIDEO_PATH` (par ex. `Simulation_1`, `Simulation_2`, …).

---

### Description détaillée des fichiers Python

### `src/stereo_globaltrack.py`

**Objectif** : script principal de **stéréovision spatio‑temporelle**.  
Il lit deux flux vidéo (gauche/droite), lance la détection YOLO, associe les détections entre caméras et entre frames, estime la position 3D de chaque objet, et produit une vidéo annotée (et éventuellement une vue de dessus temps réel).

**Fonctionnalités principales** :
- **Chargement du modèle YOLO** :
  - Chemin par défaut : variable d'environnement `MODEL_PATH` (sinon `yolov10n.pt` à la racine du projet).
- **Utilisation de la calibration stéréo** :
  - Si un fichier `.npz` de calibration est fourni (option `--calibration`), il est chargé via `StereoCalibration` et utilisé pour rectifier les images et trianguler en 3D.
  - Sinon, une calibration « simplifiée » est générée à partir de la focale et de la baseline par défaut.
- **Détection d’objets** :
  - Utilise YOLO pour détecter les objets sur chaque vue.
  - Seules les classes pertinentes pour la route sont conservées (`CLASS_MAPPING`).
  - Les seuils de confiance et d’IoU sont configurables via `.env`.
- **Triangulation 3D + tracking** :
  - Associe les détections gauche/droite via `GlobalTracker.associate_stereo(...)`.
  - Triangulation des points 3D \((X,Y,Z)\) à partir des centres bas des boîtes.
  - Mise à jour du tracker global pour obtenir des identifiants stables dans le temps.
- **Affichage et sortie vidéo** :
  - Dessine pour chaque piste :
    - **ID global**,
    - **classe** (Piéton, Cycliste, Voiture, …),
    - **distance euclidienne** à la caméra,
    - indication d’**occlusion** (épaisseur de trait réduite et label adapté).
  - Concatène les vues gauche/droite côte à côte et enregistre la vidéo de sortie (`.mp4`).
  - Ouvre une fenêtre OpenCV `Stereo View` et, si activé, une fenêtre Matplotlib de vue de dessus.

**Arguments principaux** (ligne de commande) :
- `--video-left` : chemin de la vidéo caméra gauche (par défaut `Data/<VIDEO_PATH>/gauche.mp4`).
- `--video-right` : chemin de la vidéo caméra droite (par défaut `Data/<VIDEO_PATH>/droite.mp4`).
- `--model` : chemin des poids YOLO (par défaut `MODEL_PATH` ou `yolov10n.pt`).
- `--output` : chemin de la vidéo de sortie (par défaut `runs/stereo/<VIDEO_PATH>/<VIDEO_PATH>.mp4` ou `OUTPUT_VIDEO_PATH` si défini).
- `--baseline` : baseline stéréo en mètres (par défaut `STEREO_BASELINE`).
- `--calibration` : fichier `.npz` de calibration stéréo (cf. `calibrate_stereo.py`).
- `--no-rectify` : désactive la rectification/undistortion des images.
- `--no-topdown` : désactive la vue de dessus (Matplotlib).
- `--max-lost-frames` : nombre maximal de frames d’occlusion avant suppression d’une piste.

**Utilisation de base** :

```bash
# Depuis la racine du projet (chemins par défaut depuis .env)
python src/stereo_globaltrack.py
```

**Exemples avancés** :

```bash
# Forcer une simulation spécifique sans modifier .env
VIDEO_PATH=Simulation_3 python src/stereo_globaltrack.py

# Spécifier explicitement les chemins et la calibration
python src/stereo_globaltrack.py \
  --video-left Data/Simulation_3/gauche.mp4 \
  --video-right Data/Simulation_3/droite.mp4 \
  --model yolov10n.pt \
  --calibration Data/calibration/stereo_cam.npz \
  --output runs/stereo/Simulation_3/Simulation_3.mp4
```

---

### `src/tracker.py`

**Objectif** : implémentation du **tracker global spatio‑temporel** utilisé par `stereo_globaltrack.py`.

**Principales classes** :
- **`KalmanTrack`** :
  - Représente un objet suivi dans le temps (piste).
  - Modélise un état de dimension 10 :
    - Position 3D : \((X, Y, Z)\),
    - Position image : \((c_x, c_y)\),
    - Vitesses associées : \((v_X, v_Y, v_Z, v_{cx}, v_{cy})\).
  - Utilise un **filtre de Kalman** (OpenCV) pour lisser les trajectoires et extrapoler pendant les occlusions.
  - Maintient des bounding boxes gauche/droite cohérentes avec la prédiction.

- **`GlobalTracker`** :
  - Gère l’ensemble des pistes actives (`tracks`) et l’attribution de nouveaux IDs.
  - **Association stéréo (gauche/droite)** :
    - Basée sur l’algorithme hongrois (`scipy.optimize.linear_sum_assignment`).
    - Contraintes :
      - même classe d’objet,
      - cohérence géométrique (disparité correcte, tailles compatibles),
      - écart vertical et de taille limités (`MAX_Y_DIFF`, `MAX_SIZE_DIFF`).
  - **Triangulation 3D** :
    - Utilise les matrices de projection \(P_1, P_2\) pour reconstruire les points 3D.
  - **Association temporelle (entre frames)** :
    - Prédit les états de toutes les pistes.
    - Construit une matrice de coût entre prédictions et nouvelles observations :
      - distance en pixels entre centres,
      - différence en profondeur \(Z\),
      - seuils pilotés par `MAX_PIXEL_DIST` et `MAX_DEPTH_DIST`.
    - Met à jour les pistes existantes ou crée de nouvelles pistes si nécessaire.
    - Supprime les pistes dont `lost_frames` dépasse `MAX_LOST_FRAMES`.

Les principaux hyper‑paramètres du tracker sont configurables dans `.env` (voir section dédiée ci‑dessous).

---

### `src/topdown_view.py`

**Objectif** : afficher une **vue de dessus (plan \(X-Z\))** des objets suivis.

**Caractéristiques** :
- Utilise Matplotlib en mode interactif pour afficher les coordonnées projetées \((X,Z)\) des objets suivis.
- Affiche un point coloré par piste + un label texte avec l’ID global.
- Les limites des axes sont configurables via les variables d’environnement :
  - `TOPDOWN_X_MIN`, `TOPDOWN_X_MAX`,
  - `TOPDOWN_Z_MIN`, `TOPDOWN_Z_MAX`.
- Si Matplotlib n’est pas disponible, la vue de dessus est automatiquement désactivée sans bloquer le pipeline.

---

### `src/stereo_calibration.py`

**Objectif** : encapsuler les paramètres de **calibration stéréo** et fournir des outils pour la rectification.

**Classe principale** :
- **`StereoCalibration`** (dataclass immuable) :
  - Contient :
    - Intrinsèques + distorsion : `K1`, `D1`, `K2`, `D2`,
    - Extrinsèques : `R`, `T`,
    - Taille d’image : `image_size = (w, h)`,
    - Paramètres de rectification optionnels : `R1`, `R2`, `P1`, `P2`, `Q`.
  - `load_npz(path)` :
    - Charge un fichier `.npz` contenant au minimum `K1`, `D1`, `K2`, `D2`, `R`, `T`, `image_size`.
  - `with_rectification(alpha)` :
    - Calcule (si nécessaire) `R1`, `R2`, `P1`, `P2`, `Q` via `cv2.stereoRectify`.
    - Le paramètre `alpha` (par défaut `RECTIFICATION_ALPHA`) contrôle le recadrage (0 = recadrage fort).
  - `build_rectification_maps()` :
    - Retourne les cartes pour `cv2.remap` : `(map1x, map1y, map2x, map2y)` utilisées dans `stereo_globaltrack.py`.

---

### `src/calibrate_stereo.py`

**Objectif** : outil en ligne de commande pour **estimer une calibration stéréo** à partir d’images de damier (chessboard) pour les caméras gauche/droite.

**Fonctionnement** :
- Parcourt deux dossiers d’images :
  - `--left-dir`  : images de la caméra gauche,
  - `--right-dir` : images de la caméra droite.
- Cherche automatiquement un motif de damier de taille :
  - `--cols` : nombre de coins intérieurs (colonnes),
  - `--rows` : nombre de coins intérieurs (lignes).
- Utilise `square_size_m` pour exprimer la taille réelle d’une case en mètres.
- Calibre d’abord chaque caméra individuellement, puis effectue la **calibration stéréo** complète.
- Calcule ensuite les matrices de rectification et de reprojection (`R1`, `R2`, `P1`, `P2`, `Q`).
- Sauvegarde tous les résultats dans un fichier `.npz` exploitable par `StereoCalibration`.

**Utilisation** :

```bash
python src/calibrate_stereo.py \
  --left-dir Data/calib/gauche \
  --right-dir Data/calib/droite \
  --cols 9 \
  --rows 6 \
  --square-size-m 0.025 \
  --out Data/calibration/stereo_cam.npz
```

---

### Dépendances

Les dépendances principales (déclarées dans `pyproject.toml`) sont :

- **numpy** (>=2.4.2) : calculs numériques et manipulation de tableaux.
- **ultralytics** (>=8.4.0) : framework YOLO pour la détection d'objets.
- **opencv-python** (>=4.10.0) : traitement d'images et de vidéos, lecture/écriture, affichage.
- **lapx** (>=0.5.5) : utilisé par certains trackers multi‑objets.
- **scipy** (>=1.11.0) : algorithme hongrois (`linear_sum_assignment`) pour l’association optimale.

Dépendances complémentaires recommandées :

- **python-dotenv** : chargement des variables d’environnement depuis `.env`.
- **matplotlib** : nécessaire pour activer la vue de dessus (`TopDownView`).

---

### Installation rapide

Depuis la racine du projet :

```bash
# (Optionnel mais recommandé) : créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate

# Installation des dépendances et du paquet en mode editable
pip install -e .

# Installer également les dépendances complémentaires si besoin
pip install python-dotenv matplotlib
```

Au premier lancement de `stereo_globaltrack.py`, le modèle YOLO défini par `MODEL_PATH`
sera utilisé (ou téléchargé automatiquement par Ultralytics s’il s’agit d’un modèle connu).

---

### Structure des données

- **Entrée** :
  - Dossiers de simulation dans `Data/` :
    - `Data/Simulation_1/gauche.mp4`
    - `Data/Simulation_1/droite.mp4`
    - `Data/Simulation_2/...`, etc.
- **Sortie** :
  - Vidéos annotées dans `runs/stereo/<Simulation>/` :
    - `runs/stereo/Simulation_1/Simulation_1.mp4`
  - Chemin personnalisable via `OUTPUT_VIDEO_PATH` ou l’argument `--output`.
- **Calibration** :
  - Fichiers `.npz` générés par `calibrate_stereo.py`, par exemple :
    - `Data/calibration/stereo_cam.npz`

---

### Variables d'environnement (`.env`)

Le fichier `.env` permet de régler finement le comportement du système sans modifier le code.
Les variables ci‑dessous sont lues au chargement des modules.

### Tracker (`tracker.py`)

- **KALMAN_PROCESS_NOISE** : bruit de processus du filtre de Kalman.
- **KALMAN_MEASUREMENT_NOISE** : bruit de mesure du filtre de Kalman.
- **MAX_LOST_FRAMES** : nombre max de frames consécutives sans observation avant de supprimer une piste.
- **MAX_Y_DIFF** : tolérance verticale (en pixels) pour associer les boîtes gauche/droite.
- **MAX_SIZE_DIFF** : tolérance de différence de taille (hauteur + largeur) entre boîtes gauche/droite.
- **MAX_TEMP_COST** : coût maximal pour considérer une association temporelle valide.
- **MAX_PIXEL_DIST** : distance maximale en pixels pour l’association temporelle.
- **MAX_DEPTH_DIST** : différence maximale de profondeur \(Z\) pour l’association temporelle.

### Vue de dessus (`topdown_view.py`)

- **TOPDOWN_X_MIN**, **TOPDOWN_X_MAX** : bornes de l’axe \(X\) (gauche/droite).
- **TOPDOWN_Z_MIN**, **TOPDOWN_Z_MAX** : bornes de l’axe \(Z\) (distance en avant du véhicule).

### Calibration stéréo (`stereo_calibration.py`)

- **RECTIFICATION_ALPHA** : paramètre de recadrage pour `cv2.stereoRectify` (0.0 = recadrage fort).

### Stéréovision globale (`stereo_globaltrack.py`)

- **STEREO_BASELINE** : distance entre les deux caméras (mètres), utilisée pour la calibration simplifiée.
- **STEREO_FOCAL_LENGTH** : focale en pixels (calibration simplifiée).
- **DETECTION_CONF** : seuil de confiance YOLO.
- **DETECTION_IOU** : seuil d’IoU pour la suppression des boxes.
- **FALLBACK_FPS** : FPS utilisé si la vidéo ne fournit pas d’information fiable.
- **FALLBACK_WIDTH**, **FALLBACK_HEIGHT** : résolution supposée lors de la calibration simplifiée.
- **COLORS_COUNT** : nombre de couleurs aléatoires pré‑générées pour les pistes.
- **VIDEO_PATH** : nom du sous‑dossier dans `Data/` contenant `gauche.mp4` et `droite.mp4`.
- **MODEL_PATH** : chemin du fichier de poids YOLO (ex. `yolov10n.pt`, `yolo26n.pt`, …).
- **OUTPUT_VIDEO_PATH** : chemin de sortie vidéo forcé (si vide, le chemin par défaut est utilisé).

---

### Notes techniques

- Le projet cible Python **3.11+** (cf. `pyproject.toml`).
- Le dossier `Data/` peut pointer vers un stockage externe (Google Drive, disque réseau, …) pour gérer des vidéos volumineuses.
- Les répertoires `runs/`, `Data/` et les poids de modèles (`*.pt`) sont exclus du suivi Git via `.gitignore`.
- Les scripts sont pensés pour une intégration dans un système embarqué ou semi‑embarqué, avec une priorité donnée à la **latence faible** et à la **robustesse** des chemins d'accès (utilisation de `pathlib.Path`).

