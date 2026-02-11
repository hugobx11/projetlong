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
    video_path = project_root / "Data" / "WhatsApp_Video_1.mp4"
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
        conf=0.05,  # Seuil de confiance (réduire pour détecter plus d'objets petits/loin)
        show=False,
        stream=False,
        verbose=False,
    )

    # Regrouper les classes COCO de YOLO en 4 méta‑catégories
    # piéton, véhicule, bus/camion, cycliste
    names = model.names  # ex: {0: "person", 1: "bicycle", 2: "car", ...}

    # Mapping par nom de classe YOLO -> catégorie simplifiée
    category_labels = {
        "pedestrian": "Piéton",
        "vehicle": "Véhicule",
        "bus_truck": "Bus/Camion",
        "cyclist": "Cycliste",
    }
    name_groups = {
        "pedestrian": {"person"},
        "vehicle": {"car", "motorcycle", "train", "boat"},
        "bus_truck": {"bus", "truck"},
        # Remarque: COCO n'a pas de classe "cyclist", on utilise ici "bicycle"
        "cyclist": {"bicycle"},
    }

    # Pré‑calcul d'un mapping index de classe -> catégorie
    class_to_category: dict[int, str] = {}
    for class_id, class_name in names.items():
        for cat, group in name_groups.items():
            if class_name in group:
                class_to_category[class_id] = cat
                break

    # Compter les détections pour chaque catégorie
    category_counts: dict[str, int] = {k: 0 for k in category_labels}
    for r in results:
        if r.boxes is None or r.boxes.cls is None:
            continue
        for c in r.boxes.cls.int().tolist():
            cat = class_to_category.get(int(c))
            if cat is not None:
                category_counts[cat] += 1

    print("\nCatégories détectées dans la vidéo :")
    for cat_key, label in category_labels.items():
        count = category_counts.get(cat_key, 0)
        if count > 0:
            print(f"  - {label} : {count} détections")
    print(f"\nVidéo annotée enregistrée dans : runs/detect/yolov10_video/")


if __name__ == "__main__":
    main()
