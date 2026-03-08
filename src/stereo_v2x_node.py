import argparse
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# Tes modules locaux
from tracker import GlobalTracker
from v2x_comms import V2XCommunicator

class StereoCarlaTransform:
    """
    Transforme les points 3D (X, Y, Z) issus de la stéréovision (repère caméra)
    vers les coordonnées globales du monde CARLA, en utilisant la télémétrie.
    """
    def get_transform_matrix(self, x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
        yaw_rad = np.radians(yaw_deg)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        matrix = np.array([
            [c, -s, 0, x],
            [s,  c, 0, y],
            [0,  0, 1, z],
            [0,  0, 0, 1]
        ])
        return matrix

    def camera_3d_to_world(self, x_cam: float, y_cam: float, z_cam: float, 
                           x_veh: float, y_veh: float, z_veh: float, yaw: float) -> np.ndarray:
        # 1. OpenCV (X: droite, Y: bas, Z: avant) -> Unreal Engine (X: avant, Y: droite, Z: haut)
        point_ue = np.array([z_cam, x_cam, -y_cam, 1.0])

        # 2. Application de la position/rotation du véhicule
        transform_matrix = self.get_transform_matrix(x_veh, y_veh, z_veh, yaw)
        point_world = np.dot(transform_matrix, point_ue)
        
        return point_world[:3]

def extract_detections(results, class_mapping) -> list[dict]:
    """Extrait les boîtes YOLO au format attendu par le tracker."""
    dets = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, clss, confs):
            if int(cls_id) in class_mapping:
                dets.append({"box": box, "class": int(cls_id), "conf": float(conf)})
    return dets

def main():
    parser = argparse.ArgumentParser(description="Nœud V2X avec Stéréovision")
    parser.add_argument("--vid", type=str, choices=["V1", "V2"], required=True, help="Véhicule (V1 ou V2)")
    parser.add_argument("--video-left", type=str, required=True, help="Vidéo caméra gauche (cam1)")
    parser.add_argument("--video-right", type=str, required=True, help="Vidéo caméra droite (cam2)")
    parser.add_argument("--csv", type=str, required=True, help="CSV de télémétrie")
    parser.add_argument("--model", type=str, default="yolov10n.pt", help="Modèle YOLO")
    parser.add_argument("--baseline", type=float, default=0.5, help="Baseline stéréo (m)")
    parser.add_argument("--focal", type=float, default=1200.0, help="Focale (px)")
    args = parser.parse_args()

    print(f"[{args.vid}] Démarrage du nœud Stéréo V2X...")

    # 1. Initialisations
    yolo_model = YOLO(args.model)
    df_telemetry = pd.read_csv(args.csv)
    transform_helper = StereoCarlaTransform()
    
    cap_left = cv2.VideoCapture(args.video_left)
    cap_right = cv2.VideoCapture(args.video_right)

    # Initialisation des matrices de projection simplifiées pour le tracker (comme dans stereo_globaltrack.py)
    W, H = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = np.array([[args.focal, 0, W/2], [0, args.focal, H/2], [0, 0, 1]])
    P1 = K @ np.hstack((np.eye(3), np.array([[args.baseline/2], [0], [0]])))
    P2 = K @ np.hstack((np.eye(3), np.array([[-args.baseline/2], [0], [0]])))
    
    tracker = GlobalTracker(P1, P2, max_lost_frames=5)
    communicator = V2XCommunicator(vehicle_id=args.vid)
    communicator.connect()

    class_mapping = {0: "Pieton", 1: "Cycliste", 2: "Voiture", 3: "Moto", 5: "Bus", 7: "Camion"}
    frame_idx = 0

    while cap_left.isOpened() and cap_right.isOpened():
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            break

        # 2. Récupérer la télémétrie de cette frame
        telemetry_row = df_telemetry[df_telemetry['Frame'] == frame_idx]
        if telemetry_row.empty:
            break
        
        row = telemetry_row.iloc[0]
        veh_x, veh_y, veh_z = row[f"{args.vid}_X"], row[f"{args.vid}_Y"], row[f"{args.vid}_Z"]
        veh_yaw = row[f"{args.vid}_Yaw"]

        # 3. Inférence YOLO sur les deux caméras
        res_l = yolo_model.predict(frame_l, verbose=False, conf=0.3)
        res_r = yolo_model.predict(frame_r, verbose=False, conf=0.3)
        
        dets_left = extract_detections(res_l, class_mapping)
        dets_right = extract_detections(res_r, class_mapping)

        # 4. Association Stéréo et mise à jour du Tracker
        matched_pairs = tracker.associate_stereo(dets_left, dets_right)
        tracker.update_tracks(matched_pairs)

        # 5. Transformation en coordonnées Mondiales pour l'envoi V2X
        local_objects_for_v2x = []
        for track_id, track in tracker.tracks.items():
            if track.lost_frames == 0:  # Seulement les objets vus à cette frame
                state = track.current_state
                x_cam, y_cam, z_cam = state["X"], state["Y"], state["Z"]
                
                # Conversion en coordonnées CARLA
                world_xyz = transform_helper.camera_3d_to_world(x_cam, y_cam, z_cam, veh_x, veh_y, veh_z, veh_yaw)
                
                local_objects_for_v2x.append({
                    "id": track_id,
                    "class": track.class_id,
                    "position": {"x": float(world_xyz[0]), "y": float(world_xyz[1]), "z": float(world_xyz[2])},
                    "confidence": 0.8  # Confiance arbitraire ou issue de YOLO
                })

                # Affichage sur la frame gauche
                box = track.box_l
                cv2.rectangle(frame_l, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(frame_l, f"ID:{track_id} Z:{z_cam:.1f}m", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 6. Partage Réseau V2X
        communicator.publish_perceptions(local_objects_for_v2x)

        # 7. Réception V2X et Fusion
        v2x_objects = communicator.get_and_clear_v2x_objects()
        
        # (La méthode fuse_v2x_observations mettra à jour les pistes ou créera des pistes coopératives)
        if hasattr(tracker, 'fuse_v2x_observations'):
            tracker.fuse_v2x_observations(v2x_objects)

        # Affichage des objets V2X (en texte sur l'écran pour debug)
        y_offset = 30
        cv2.putText(frame_l, "Objets V2X distants recus :", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for obj in v2x_objects:
            y_offset += 25
            cv2.putText(frame_l, f"ID:{obj['id']} Classe:{obj['class']} -> X:{obj['position']['x']:.1f}, Y:{obj['position']['y']:.1f}", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(f"Stereo V2X - {args.vid}", frame_l)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    communicator.disconnect()

if __name__ == "__main__":
    main()