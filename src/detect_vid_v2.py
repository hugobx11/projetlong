#!/usr/bin/env python3
"""
Détection d'entités simplifiée avec caractéristiques (Couleur) pour RoadEye Guardian.
Utilise YOLOv10n pour la performance et OpenCV pour l'analyse de couleur.
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Mapping des classes COCO vers nos catégories simplifiées (Français)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
CLASS_MAPPING = {
    0: "Pieton",
    1: "Cycliste",
    2: "Voiture",
    3: "Cycliste",  # On regroupe moto et vélo
    5: "Bus",
    7: "Camion"
}

# Couleurs de référence en HSV (Hue, Saturation, Value)
# Ranges approximatifs pour une classification simple
COLORS_HSV = {
    "Rouge": [(0, 10), (170, 180)],
    "Orange": [(11, 25)],
    "Jaune": [(26, 34)],
    "Vert": [(35, 85)],
    "Bleu": [(86, 125)],
    "Violet": [(126, 169)]
}

def get_dominant_color(roi: np.ndarray) -> str:
    """
    Détermine la couleur dominante d'une région d'intérêt (ROI).
    Utilise l'espace HSV et le centre de l'image pour éviter le fond.
    """
    if roi.size == 0:
        return "Inconnue"

    # On prend le centre de l'objet pour éviter de capter le bitume/fond
    h, w, _ = roi.shape
    center_crop = roi[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    
    if center_crop.size == 0:
        return "Inconnue"

    # Conversion en HSV
    hsv_roi = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    
    # Calcul des moyennes
    avg_h = np.mean(hsv_roi[:, :, 0])
    avg_s = np.mean(hsv_roi[:, :, 1])
    avg_v = np.mean(hsv_roi[:, :, 2])

    # 1. Analyse des nuances de gris (Blanc, Noir, Gris)
    # Si saturation basse ou valeur très basse/haute
    if avg_s < 40: # Peu de couleur -> Gris/Blanc/Noir
        if avg_v < 50: return "Noir"
        if avg_v > 200: return "Blanc"
        return "Gris"
    
    if avg_v < 30: # Très sombre -> Noir (même si saturé)
        return "Noir"

    # 2. Analyse de la teinte (Hue)
    for color_name, ranges in COLORS_HSV.items():
        for r in ranges:
            if len(r) == 2: # Range unique (ex: Orange)
                if r[0] <= avg_h <= r[1]:
                    return color_name
            elif len(r) == 1: # Cas spécial
                pass 
            
    # Cas du rouge qui boucle (0-10 et 170-180)
    if (0 <= avg_h <= 10) or (170 <= avg_h <= 180):
        return "Rouge"

    return "Autre"

def main() -> None:
    project_root = Path(__file__).resolve().parent
    video_path = project_root / "Data" / "onboard_driving_New_York.mp4"
    output_path = project_root / "runs" / "detect" / "output_guardian.mp4"
    
    # Création du dossier de sortie
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    # Chargement modèle léger
    print("Chargement de YOLOv10n...")
    model = YOLO("yolov10n.pt")

    # Ouverture vidéo
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Codec vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Traitement de la vidéo (Sortie: {output_path})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inférence (classes filtrées pour la rapidité)
        results = model.predict(frame, conf=0.4, verbose=False, classes=list(CLASS_MAPPING.keys()))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordonnées
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                
                # Nom simplifié
                label = CLASS_MAPPING.get(cls_id, "Objet")
                
                # Extraction couleur
                roi = frame[y1:y2, x1:x2]
                color = get_dominant_color(roi)

                # Affichage graphique
                # Rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Texte
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