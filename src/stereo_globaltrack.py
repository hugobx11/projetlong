#!/usr/bin/env python3
"""
Estimation de distance par stéréovision spatio-temporelle.
Remplace ByteTrack par un GlobalTracker personnalisé utilisant l'algorithme Hongrois
et les contraintes épipolaires pour maintenir des IDs cohérents entre les caméras.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

CLASS_MAPPING = {
    0: "Pieton", 1: "Cycliste", 2: "Voiture", 
    3: "Moto", 5: "Bus", 7: "Camion"
}

# ---------------------------------------------------------------------------
# 1. MODULE DE TRACKING GLOBAL SPATIO-TEMPOREL
# ---------------------------------------------------------------------------
class KalmanTrack:
    """
    Représente une piste (track) suivie dans le temps.

    Cette classe encapsule :
    - l'identifiant unique de la piste (`id`),
    - la classe d'objet détectée (piéton, voiture, etc.),
    - l'état 3D + 2D de l'objet modélisé par un filtre de Kalman,
    - les bounding boxes projetées dans chaque vue (gauche / droite),
    - un compteur d'images où l'objet n'a pas été observé (`lost_frames`),
      utilisé pour gérer les occlusions temporaires.
    """
    def __init__(self, track_id, obs):
        self.id = track_id
        self.class_id = obs['class']
        self.lost_frames = 0
        self.box_l = obs['box_l']
        self.box_r = obs['box_r']

        # Filtre de Kalman : 10 variables d'état (X,Y,Z, cx,cy, vX,vY,vZ, vcx,vcy), 5 mesures
        # On suit à la fois la position 3D du point et sa projection (cx, cy) dans l'image.
        self.kf = cv2.KalmanFilter(10, 5)
        
        # Matrice de transition (A) : Modèle à vitesse constante (Position = Position + Vitesse)
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        for i in range(5):
            self.kf.transitionMatrix[i, i+5] = 1.0 
            
        # Matrice de mesure (H)
        self.kf.measurementMatrix = np.zeros((5, 10), dtype=np.float32)
        for i in range(5):
            self.kf.measurementMatrix[i, i] = 1.0
            
        # Bruit de processus (confiance en la prédiction) et Bruit de mesure (confiance en YOLO)
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(5, dtype=np.float32) * 0.5 # Plus c'est haut, plus on lisse la 3D
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)
        
        # Initialisation de l'état : on part des mesures observées (3D + centre image)
        # et on initialise les vitesses à zéro.
        state = np.array([obs['X'], obs['Y'], obs['Z'], obs['cx'], obs['cy'], 0, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = state.reshape(-1, 1)
        self.kf.statePre = state.reshape(-1, 1)

        self.current_state = {
            'X': obs['X'], 'Y': obs['Y'], 'Z': obs['Z'], 'cx': obs['cx'], 'cy': obs['cy']
        }

    def predict(self):
        """
        Étape de prédiction du filtre de Kalman.

        - Fait avancer l'état dans le temps en utilisant le modèle dynamique.
        - Met à jour la position 3D et la position projetée (cx, cy).
        - Déplace les boîtes gauche/droite en cohérence avec la nouvelle position (cx, cy),
          ce qui permet de continuer à approximer la localisation pendant les occlusions.
        """
        old_cx, old_cy = self.current_state['cx'], self.current_state['cy']
        pred = self.kf.predict()
        
        self.current_state['X'] = float(pred[0, 0])
        self.current_state['Y'] = float(pred[1, 0])
        self.current_state['Z'] = float(pred[2, 0])
        self.current_state['cx'] = float(pred[3, 0])
        self.current_state['cy'] = float(pred[4, 0])
        
        # Décalage visuel des boîtes pendant l'occlusion pour suivre la prédiction :
        # on translate les bounding boxes de la même quantité que le déplacement prédit du centre.
        dx = self.current_state['cx'] - old_cx
        dy = self.current_state['cy'] - old_cy
        
        self.box_l = [self.box_l[0]+dx, self.box_l[1]+dy, self.box_l[2]+dx, self.box_l[3]+dy]
        self.box_r = [self.box_r[0]+dx, self.box_r[1]+dy, self.box_r[2]+dx, self.box_r[3]+dy]
        
        return self.current_state

    def update(self, obs):
        """
        Étape de mise à jour (correction) du filtre de Kalman.

        Prend en entrée une nouvelle observation (issue de YOLO + triangulation),
        met à jour l'état interne du Kalman, les boîtes gauche/droite et réinitialise
        le compteur de frames perdues.
        """
        meas = np.array([obs['X'], obs['Y'], obs['Z'], obs['cx'], obs['cy']], dtype=np.float32)
        self.kf.correct(meas.reshape(-1, 1))
        self.box_l = obs['box_l']
        self.box_r = obs['box_r']
        self.lost_frames = 0
        
        post = self.kf.statePost
        self.current_state['X'] = float(post[0, 0])
        self.current_state['Y'] = float(post[1, 0])
        self.current_state['Z'] = float(post[2, 0])
        self.current_state['cx'] = float(post[3, 0])
        self.current_state['cy'] = float(post[4, 0])

class GlobalTracker:
    """
    Gère l'ensemble des pistes globales (multi-objets, multi-caméras, multi-frames).

    Rôles principaux :
    - associer les détections gauche/droite (stéréo) pour obtenir des couples cohérents,
    - estimer la position 3D de chaque couple par triangulation,
    - associer ces observations 3D aux pistes existantes dans le temps,
    - créer de nouvelles pistes quand un nouvel objet apparaît,
    - supprimer les pistes devenues trop anciennes (trop de frames perdues).
    """
    def __init__(self, P1, P2, max_lost_frames=5):
        self.P1 = P1
        self.P2 = P2
        self.tracks = {} 
        self.next_id = 1
        self.max_lost_frames = max_lost_frames

    def associate_stereo(self, dets_left, dets_right, max_y_diff=30, max_size_diff=50):
        """
        Associe les détections de la caméra gauche avec celles de la caméra droite.

        Critères d'association :
        - même classe d'objet des deux côtés,
        - ordre horizontal cohérent (caméra 2 à droite → disparité correcte),
        - écart vertical (Y) limité pour respecter la contrainte épipolaire,
        - tailles de boîtes semblables.

        Retourne une liste de paires (det_l, det_r) déjà "matchées".
        """
        if len(dets_left) == 0 or len(dets_right) == 0:
            return []

        MAX_COST = 1e5  # Coût initial élevé : seuls les couples valides verront ce coût réduit
        cost_matrix = np.full((len(dets_left), len(dets_right)), MAX_COST)

        for i, det_l in enumerate(dets_left):
            for j, det_r in enumerate(dets_right):
                # Contrainte de classe
                if det_l['class'] != det_r['class']:
                    continue

                xl_c, yl_c = (det_l['box'][0] + det_l['box'][2]) / 2, (det_l['box'][1] + det_l['box'][3]) / 2
                hl, wl = det_l['box'][3] - det_l['box'][1], det_l['box'][2] - det_l['box'][0]

                xr_c, yr_c = (det_r['box'][0] + det_r['box'][2]) / 2, (det_r['box'][1] + det_r['box'][3]) / 2
                hr, wr = det_r['box'][3] - det_r['box'][1], det_r['box'][2] - det_r['box'][0]

                # Contrainte d'ordre horizontal :
                # La caméra 2 est à droite → la projection de l'objet devrait être plus à gauche
                # dans l'image de droite (disparité positive).
                if xr_c > xl_c:
                    continue

                # Contrainte épipolaire (approximation) :
                # on impose que les coordonnées verticales des centres soient proches.
                y_diff = abs(yl_c - yr_c)
                if y_diff > max_y_diff:
                    continue

                # Contrainte de taille : on évite d'associer des objets aux dimensions trop différentes.
                size_diff = abs(hl - hr) + abs(wl - wr)
                if size_diff > max_size_diff:
                    continue

                cost_matrix[i, j] = y_diff + (size_diff * 0.5)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] != np.inf:
                matched_pairs.append((dets_left[r], dets_right[c]))
                
        return matched_pairs

    def estimate_3d_position(self, det_l, det_r):
        """Utilise la triangulation d'OpenCV pour calculer les coordonnées 3D (X, Y, Z)."""
        def get_bottom_center(box):
            return np.array([[(box[0] + box[2]) / 2], [box[3]]]) # Centre bas
            
        pt1_f = get_bottom_center(det_l['box']).astype(float)
        pt2_f = get_bottom_center(det_r['box']).astype(float)

        point_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_f, pt2_f)
        point_3d = point_4d[:3] / point_4d[3]
        return point_3d.flatten() # Retourne [X, Y, Z]
    
    def update_tracks(self, matched_pairs):
        """
        Met à jour l'ensemble des pistes à partir des couples stéréo associés.

        Étapes :
        1. Projection stéréo → estimation 3D pour chaque couple (X, Y, Z, cx, cy).
        2. Prédiction de tous les KalmanTrack (étape `predict`).
        3. Construction d'une matrice de coût entre pistes existantes et nouvelles observations.
        4. Association optimale via l'algorithme Hongrois.
        5. Mise à jour des pistes associées, création de nouvelles pistes pour les non-associés.
        6. Incrément du compteur `lost_frames` pour les pistes non observées et nettoyage des pistes obsolètes.
        """
        current_frame_observations = []
        for det_l, det_r in matched_pairs:
            X, Y, Z = self.estimate_3d_position(det_l, det_r)
            cx = (det_l['box'][0] + det_l['box'][2]) / 2
            cy = (det_l['box'][1] + det_l['box'][3]) / 2
            current_frame_observations.append({
                'class': det_l['class'], 'X': X, 'Y': Y, 'Z': Z, 'cx': cx, 'cy': cy,
                'box_l': det_l['box'], 'box_r': det_r['box']
            })

        track_ids = list(self.tracks.keys())

        # 2. On fait avancer TOUS les filtres de Kalman (Prédiction de l'état futur)
        predicted_states = {tid: self.tracks[tid].predict() for tid in track_ids}

        # 3. Association des pistes existantes avec les nouvelles observations
        MAX_TEMP_COST = 1000.0
        cost_matrix = np.full((len(track_ids), len(current_frame_observations)), MAX_TEMP_COST)

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            pred = predicted_states[tid]
            for j, obs in enumerate(current_frame_observations):
                if track.class_id != obs['class']: continue
                
                # On compare l'observation avec la PREDICTION (et non l'ancienne position)
                dist_pixel = np.sqrt((pred['cx'] - obs['cx'])**2 + (pred['cy'] - obs['cy'])**2)
                dist_z = abs(pred['Z'] - obs['Z'])

                # On n'autorise une association que si la prédiction et l'observation sont proches
                # en image (dist_pixel) ET en profondeur (dist_z).
                if dist_pixel < 200.0 and dist_z < 15.0:
                    cost_matrix[i, j] = dist_pixel

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_obs = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < MAX_TEMP_COST:
                tid = track_ids[r]
                obs = current_frame_observations[c]
                self.tracks[tid].update(obs) # Corrige la trajectoire avec YOLO
                assigned_obs.add(c)

        # 4. Gérer les non assignés (Occlusion) et créations
        for r in range(len(track_ids)):
            if r not in row_ind or cost_matrix[r, col_ind[list(row_ind).index(r)]] == MAX_TEMP_COST:
                # L'objet n'a pas été observé : on incrémente son compteur d'absence.
                # Il continue néanmoins à avancer grâce à sa vitesse prédite.
                self.tracks[track_ids[r]].lost_frames += 1

        for j, obs in enumerate(current_frame_observations):
            if j not in assigned_obs:
                self.tracks[self.next_id] = KalmanTrack(self.next_id, obs)
                self.next_id += 1

        self.tracks = {k: v for k, v in self.tracks.items() if v.lost_frames < self.max_lost_frames}

# ---------------------------------------------------------------------------
# 2. APPLICATION PRINCIPALE DE VISION STÉRÉO
# ---------------------------------------------------------------------------
class StereoDepthEstimator:
    def __init__(self, model_path: str, baseline: float = 0.5, focal_length: float = 1200.0):
        self.model = YOLO(model_path)
        
        W, H = 1920, 1080
        self.K = np.array([
            [focal_length, 0, W/2],
            [0, focal_length, H/2],
            [0, 0, 1]
        ])

        self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        R = np.eye(3)
        T = np.array([[-baseline], [0], [0]]) 
        self.P2 = self.K @ np.hstack((R, T))
        
        # Initialisation de notre nouveau tracker
        self.tracker = GlobalTracker(self.P1, self.P2)

    def extract_detections(self, results):
        """Convertit le format YOLO en liste de dictionnaires pour le tracker."""
        dets = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes, clss, confs):
                if int(cls_id) in CLASS_MAPPING: # Filtrage des classes pertinentes
                    dets.append({'box': box, 'class': int(cls_id), 'conf': float(conf)})
        return dets

    def process_videos(self, video1_path: Path, video2_path: Path, output_path: Path):
        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))

        if not cap1.isOpened() or not cap2.isOpened():
            raise IOError("Impossible d'ouvrir l'une des vidéos.")

        fps = cap1.get(cv2.CAP_PROP_FPS)
        w, h = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_size = (w * 2, h)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
        print(f"Enregistrement commencé : {output_path}")

        cv2.namedWindow("Stereo View", cv2.WINDOW_NORMAL)

        # Générateur de couleurs pour les IDs
        colors = np.random.randint(0, 255, size=(1000, 3), dtype=int)

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # 1. Détection indépendante (Sans ByteTrack, on utilise predict)
            # conf=0.3 : On ne garde que les détections dont l'IA est sûre à 30% minimum
            # iou=0.4  : Si deux boîtes se chevauchent à plus de 40%, on supprime la moins probable
            res1 = self.model.predict(frame1, verbose=False, conf=0.3, iou=0.4)
            res2 = self.model.predict(frame2, verbose=False, conf=0.3, iou=0.4)

            # 2. Formatage des données
            dets_left = self.extract_detections(res1)
            dets_right = self.extract_detections(res2)

            # 3. Association Spatiale Stéréo
            matched_pairs = self.tracker.associate_stereo(dets_left, dets_right)

            # 4. Association Temporelle et calcul 3D
            self.tracker.update_tracks(matched_pairs)

            # 5. Dessin et Affichage unifié
            for track_id, track in self.tracker.tracks.items():
                color = tuple(int(c) for c in colors[track_id % len(colors)])
                label_name = CLASS_MAPPING.get(track.class_id, "Inconnu")
                state = track.current_state
                distance = np.sqrt(state['X']**2 + state['Y']**2 + state['Z']**2)
                
                # Si l'objet est en occlusion, on l'indique textuellement
                status = " (Cache)" if track.lost_frames > 0 else ""
                text = f"ID:{track_id} {label_name}{status} {distance:.1f}m"

                # Pour éviter que la boîte ne soit trop "ferme" si elle est inventée par le filtre, 
                # on peut la dessiner avec une épaisseur plus fine pendant l'occlusion
                thickness = 1 if track.lost_frames > 0 else 2

                box_l = track.box_l
                cv2.rectangle(frame1, (int(box_l[0]), int(box_l[1])), (int(box_l[2]), int(box_l[3])), color, thickness)
                cv2.putText(frame1, text, (int(box_l[0]), int(box_l[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                box_r = track.box_r
                cv2.rectangle(frame2, (int(box_r[0]), int(box_r[1])), (int(box_r[2]), int(box_r[3])), color, thickness)
                cv2.putText(frame2, text, (int(box_r[0]), int(box_r[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            h1, w1 = frame1.shape[:2]
            frame2_resized = cv2.resize(frame2, (w1, h1))
            vis = np.hstack((frame1, frame2_resized))

            out.write(vis)
            cv2.imshow("Stereo View", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap1.release()
        cap2.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    root = Path(__file__).resolve().parent.parent
    video1 = root / "Data" / "CARLA" / "Simulation statique 1" / "video_cam1_gauche.mp4" 
    video2 = root / "Data" / "CARLA" / "Simulation statique 1" / "video_cam2_droite.mp4"
    model_pt = root / "yolov10n.pt"

    output_dir = root / "runs" / "stereo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stereo_globaltrack_result.mp4"

    if not video1.exists() or not video2.exists():
        print(f"ATTENTION: Vidéos introuvables dans {root}/Data/")
        return

    # Instanciation de l'estimateur
    estimator = StereoDepthEstimator(str(model_pt), baseline=0.5)
    estimator.process_videos(video1, video2, output_path)

if __name__ == "__main__":
    main()