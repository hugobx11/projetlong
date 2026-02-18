#!/usr/bin/env python3
"""
Estimation de distance par stéréovision (2 caméras) avec YOLOv10 et ByteTrack.
Nécessite deux vidéos synchronisées de la même scène.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class StereoDepthEstimator:
    def __init__(self, model_path: str, baseline: float = 0.5, focal_length: float = 1200.0):
        """
        :param model_path: Chemin vers le modèle YOLO
        :param baseline: Distance entre les deux caméras (en mètres)
        :param focal_length: Focale (en pixels). Dépend de la résolution et de l'angle de vue.
        """
        # Chargement du modèle
        self.model = YOLO(model_path)
        
        # --- CALIBRATION SIMULÉE (A REMPLACER PAR VOS VALEURS RÉELLES) ---
        # Matrice Intrinsèque (K) supposée identique pour les 2 caméras
        # Centre optique (cx, cy) supposé au centre de l'image (ex: 1920x1080)
        W, H = 1920, 1080
        self.K = np.array([
            [focal_length, 0, W/2],
            [0, focal_length, H/2],
            [0, 0, 1]
        ])

        # Matrices de Projection (P = K * [R|T])
        # Caméra 1 : Référence (Position 0,0,0, Rotation identité)
        self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # Caméra 2 : Décalée de 'baseline' sur l'axe X (droite)
        # T = [-baseline, 0, 0] car la caméra 2 voit le monde décalé
        R = np.eye(3)
        T = np.array([[-baseline], [0], [0]]) 
        self.P2 = self.K @ np.hstack((R, T))

    def get_center_point(self, box):
        """Retourne le centre bas de la bounding box (point de contact au sol)"""
        x1, y1, x2, y2 = box
        return np.array([[(x1 + x2) / 2], [y2]]) # Shape (2, 1)

    def process_videos(self, video1_path: Path, video2_path: Path):
        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))

        if not cap1.isOpened() or not cap2.isOpened():
            raise IOError("Impossible d'ouvrir l'une des vidéos.")

        # Création des fenêtres
        cv2.namedWindow("Stereo View", cv2.WINDOW_NORMAL)

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # 1. Détection & Tracking (YOLO + ByteTrack)
            # On utilise le tracker pour lisser les detections, mais l'association
            # Stereo se fait frame par frame ici.
            res1 = self.model.track(frame1, persist=True, verbose=False ,conf=0.05, tracker="bytetrack.yaml")
            res2 = self.model.track(frame2, persist=True, verbose=False ,conf=0.05, tracker="bytetrack.yaml")

            # 2. Extraction des points d'intérêt (Centres bas des boites)
            points_cam1 = []
            points_cam2 = []
            labels_cam1 = []
            
            # Récupération des boites Cam 1
            #if res1[0].boxes.id is not None:
            if res1[0].boxes is not None:
                boxes1 = res1[0].boxes.xyxy.cpu().numpy()
                clss1 = res1[0].boxes.cls.cpu().numpy()
                for box, cls_id in zip(boxes1, clss1):
                    pt = self.get_center_point(box)
                    points_cam1.append(pt)
                    labels_cam1.append(self.model.names[int(cls_id)])
                    # Dessin Cam 1
                    cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

            # Récupération des boites Cam 2
            #if res2[0].boxes.id is not None:
            if res2[0].boxes is not None:
                boxes2 = res2[0].boxes.xyxy.cpu().numpy()
                for box in boxes2:
                    pt = self.get_center_point(box)
                    points_cam2.append(pt)
                    # Dessin Cam 2
                    cv2.rectangle(frame2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    

            # Trier points_cam1 par ordre croissant du premier élément avec fonctions natives
            indices_triés1 = np.argsort([pt[0, 0] for pt in points_cam1])
            points_cam1 = [points_cam1[i] for i in indices_triés1]
            labels_cam1 = [labels_cam1[i] for i in indices_triés1]
            print(f"nb entités cam1: {len(points_cam1)}")

            indices_triés2 = np.argsort([pt[0, 0] for pt in points_cam2])
            points_cam2 = [points_cam2[i] for i in indices_triés2]
            print(f"nb entités cam2: {len(points_cam2)}")


            # 3. Matching Naïf (Le plus grand défi en stéréovision)
            # Ici, on suppose qu'il n'y a qu'un seul objet principal ou on matche par ordre.
            # Pour un vrai projet : il faut matcher par Similarité Visuelle (ReID) ou contrainte épipolaire.
            
            min_len = min(len(points_cam1), len(points_cam2))
            for i in range(min_len):
                pt1 = points_cam1[i]
                pt2 = points_cam2[i]
                label = labels_cam1[i]
                # 4. Triangulation
                # Convertir en format accepté par opencv (2, N) float
                pt1_f = pt1.astype(float)
                pt2_f = pt2.astype(float)

                # Output: 4D point (X, Y, Z, W) en coordonnées homogènes
                point_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_f, pt2_f)
                
                # Conversion en 3D cartésien (diviser par W)
                point_3d = point_4d[:3] / point_4d[3]
                
                X, Y, Z = point_3d.flatten()
                
                # Distance Euclidienne depuis la Caméra 1
                distance = np.sqrt(X**2 + Y**2 + Z**2)

                # Affichage
                info_text = f"{label}: {distance:.2f}m (X:{X:.1f}, Z:{Z:.1f})"
                cv2.putText(frame1, info_text, (int(pt1[0,0]), int(pt1[1,0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                print(f"label: {label}{i}, distance: {distance}")

            # Concaténation pour affichage
            # Redimensionner pour coller côte à côte
            h1, w1 = frame1.shape[:2]
            frame2_resized = cv2.resize(frame2, (w1, h1))
            vis = np.hstack((frame1, frame2_resized))

            cv2.imshow("Stereo View", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

def main():
    root = Path(__file__).resolve().parent.parent
    # Adaptez les noms de vos vidéos ici
    video1 = root / "Data" / "Simulation statique 1" / "video_cam1_gauche.mp4" 
    video2 = root / "Data" / "Simulation statique 1" / "video_cam2_droite.mp4"
    model_pt = root / "yolov10n.pt"

    # Exemple : Caméras écartées de 50cm (0.5m)
    estimator = StereoDepthEstimator(str(model_pt), baseline=0.5)
    
    # Création de fichiers dummy si absents pour éviter le crash immédiat du test
    if not video1.exists() or not video2.exists():
        print(f"ATTENTION: Vidéos introuvables dans {root}/Data/")
        print("Veuillez fournir deux vidéos : video_left.mp4 et video_right.mp4")
        return

    estimator.process_videos(video1, video2)

if __name__ == "__main__":
    main()