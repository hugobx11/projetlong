#!/usr/bin/env python3
"""
Détection d'entités (personnes, véhicules, etc.) dans une vidéo avec YOLOv10.
Lit la vidéo dans Data/ et enregistre la vidéo annotée + un résumé des détections.
"""

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    # Chemins
    project_root = Path(__file__).resolve().parent
    video_path = project_root / "Data" / "onboard_driving_New_York.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    # Charger YOLOv10 (nano = rapide, ou yolov10s/yolov10m pour plus de précision)
    model = YOLO("yolov10n.pt")

    # Détection sur la vidéo
    results = model.predict(
        source=str(video_path),
        save=True,
        project=str(project_root / "runs" / "detect"),
        name="yolov10_video",
        exist_ok=True,
        conf=0.25,  # Seuil de confiance (réduire pour détecter plus d'objets petits/loin)
        show=False,
        stream=False,
    )

    # Résumé des classes détectées
    all_classes: set[int] = set()
    for r in results:
        if r.boxes is not None and r.boxes.cls is not None:
            all_classes.update(r.boxes.cls.int().tolist())

    names = model.names
    print("\nEntités détectées dans la vidéo :")
    for c in sorted(all_classes):
        print(f"  - {names[c]} (classe {c})")
    print(f"\nVidéo annotée enregistrée dans : runs/detect/yolov10_video/")


if __name__ == "__main__":
    main()
