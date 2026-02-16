#!/usr/bin/env python3
"""
Détection d'entités simplifiée avec caractéristiques (Couleur) pour RoadEye Guardian.
Utilise YOLOv10n pour la performance et OpenCV pour l'analyse de couleur.
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Mapping des classes COCO vers nos catégories simplifiées
CLASS_MAPPING = {
    0: "Pieton",
    1: "Cycliste",
    2: "Voiture",
    3: "Moto",
    5: "Bus",
    7: "Camion"
}

# Couleurs de référence en HSV
COLORS_HSV = {
    "Rouge": [(0, 10), (170, 180)],
    "Orange": [(11, 25)],
    "Jaune": [(26, 34)],
    "Vert": [(35, 85)],
    "Bleu": [(86, 125)],
    "Violet": [(126, 169)]
}

def get_dominant_color(roi: np.ndarray) -> str:
    if roi.size == 0:
        return "Inconnue"

    h, w, _ = roi.shape
    center_crop = roi[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    
    if center_crop.size == 0:
        return "Inconnue"

    hsv_roi = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    
    avg_h = np.mean(hsv_roi[:, :, 0])
    avg_s = np.mean(hsv_roi[:, :, 1])
    avg_v = np.mean(hsv_roi[:, :, 2])

    if avg_s < 40:
        if avg_v < 50: return "Noir"
        if avg_v > 200: return "Blanc"
        return "Gris"
    
    if avg_v < 30:
        return "Noir"

    for color_name, ranges in COLORS_HSV.items():
        for r in ranges:
            if len(r) == 2:
                if r[0] <= avg_h <= r[1]:
                    return color_name
            elif len(r) == 1:
                pass 
            
    if (0 <= avg_h <= 10) or (170 <= avg_h <= 180):
        return "Rouge"

    return "Autre"

def main() -> None:
    # --- CORRECTION CHEMIN ---
    project_root = Path(__file__).resolve().parent.parent
    
    # Assurez-vous que cette vidéo existe bien dans votre dossier Data
    video_path = project_root / "Data" / "onboard_driving_New_York.mp4"
    output_path = project_root / "runs" / "detect" / "res_detect_vid.mp4"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    print("Chargement de YOLOv10n...")
    model_path = project_root / "yolov10n.pt"
    model = YOLO(model_path if model_path.exists() else "yolov10n.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
         raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Traitement de la vidéo (Sortie: {output_path})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.4, verbose=False, classes=list(CLASS_MAPPING.keys()))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                
                label = CLASS_MAPPING.get(cls_id, "Objet")
                
                roi = frame[y1:y2, x1:x2]
                color = get_dominant_color(roi)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text = f"{label} | {color}"
                t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("Terminé.")

if __name__ == "__main__":
    main()