#!/usr/bin/env python3
"""
Estimation de distance par stéréovision spatio-temporelle.

Boucle d'application principale : lecture des deux vidéos, détection YOLO,
association stéréo + temporelle via GlobalTracker, affichage OpenCV
et vue de dessus (X-Z) optionnelle.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import GlobalTracker
from topdown_view import TopDownView
from stereo_calibration import StereoCalibration

load_dotenv()

STEREO_BASELINE_DEFAULT = float(os.getenv("STEREO_BASELINE", "0.5"))
STEREO_FOCAL_LENGTH_DEFAULT = float(os.getenv("STEREO_FOCAL_LENGTH", "1200.0"))
MAX_LOST_FRAMES_DEFAULT = int(os.getenv("MAX_LOST_FRAMES", "5"))

DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.3"))
DETECTION_IOU = float(os.getenv("DETECTION_IOU", "0.4"))

FALLBACK_FPS = float(os.getenv("FALLBACK_FPS", "25.0"))
FALLBACK_WIDTH = int(os.getenv("FALLBACK_WIDTH", "1920"))
FALLBACK_HEIGHT = int(os.getenv("FALLBACK_HEIGHT", "1080"))

COLORS_COUNT = int(os.getenv("COLORS_COUNT", "1000"))

CLASS_MAPPING = {
    0: "Pieton",
    1: "Cycliste",
    2: "Voiture",
    3: "Moto",
    5: "Bus",
    7: "Camion",
}


class StereoDepthEstimator:
    def __init__(
        self,
        model_path: str,
        baseline: float = STEREO_BASELINE_DEFAULT,
        focal_length: float = STEREO_FOCAL_LENGTH_DEFAULT,
        max_lost_frames: int = MAX_LOST_FRAMES_DEFAULT,
        calibration_path: Path | None = None,
        rectify: bool = True,
    ):
        self.model = YOLO(model_path)

        self._rectify = rectify
        self._maps = None

        if calibration_path is not None:
            calib = StereoCalibration.load_npz(calibration_path)
            if self._rectify:
                calib = calib.with_rectification(alpha=0.0)
                self.P1 = calib.P1
                self.P2 = calib.P2
                self._maps = calib.build_rectification_maps()
            else:
                # Sans rectification, on peut quand même trianguler avec P1/P2 si présents,
                # sinon l'utilisateur doit rectifier en amont.
                calib = calib.with_rectification(alpha=0.0)
                self.P1 = calib.P1
                self.P2 = calib.P2
        else:
            # Fallback "simplifié" si pas de calibration fournie
            W, H = FALLBACK_WIDTH, FALLBACK_HEIGHT
            self.K = np.array(
                [
                    [focal_length, 0, W / 2],
                    [0, focal_length, H / 2],
                    [0, 0, 1],
                ]
            )
            self.P1 = self.K @ np.hstack(
                (np.eye(3), np.array([[baseline / 2], [0], [0]]))
            )
            R = np.eye(3)
            T = np.array([[-baseline / 2], [0], [0]])
            self.P2 = self.K @ np.hstack((R, T))

        self.tracker = GlobalTracker(self.P1, self.P2, max_lost_frames=max_lost_frames)

    def extract_detections(self, results) -> list[dict]:
        dets: list[dict] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, clss, confs):
                if int(cls_id) in CLASS_MAPPING:
                    dets.append(
                        {"box": box, "class": int(cls_id), "conf": float(conf)}
                    )
        return dets

    def process_videos(
        self,
        video1_path: Path,
        video2_path: Path,
        output_path: Path,
        show_topdown: bool = True,
    ) -> None:
        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))

        if not cap1.isOpened() or not cap2.isOpened():
            raise IOError("Impossible d'ouvrir l'une des vidéos.")

        fps = cap1.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_size = (w * 2, h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
        print(f"Enregistrement commencé : {output_path}")

        cv2.namedWindow("Stereo View", cv2.WINDOW_NORMAL)

        colors = np.random.randint(0, 255, size=(COLORS_COUNT, 3), dtype=int)

        topdown = TopDownView() if show_topdown else None

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Rectification / undistortion (si calibration fournie)
            if self._maps is not None:
                map1x, map1y, map2x, map2y = self._maps
                frame1 = cv2.remap(frame1, map1x, map1y, interpolation=cv2.INTER_LINEAR)
                frame2 = cv2.remap(frame2, map2x, map2y, interpolation=cv2.INTER_LINEAR)

            res1 = self.model.predict(
                frame1, verbose=False, conf=DETECTION_CONF, iou=DETECTION_IOU
            )
            res2 = self.model.predict(
                frame2, verbose=False, conf=DETECTION_CONF, iou=DETECTION_IOU
            )

            dets_left = self.extract_detections(res1)
            dets_right = self.extract_detections(res2)

            matched_pairs = self.tracker.associate_stereo(dets_left, dets_right)

            self.tracker.update_tracks(matched_pairs)

            if topdown is not None:
                topdown.update(self.tracker.tracks, colors)

            for track_id, track in self.tracker.tracks.items():
                color = tuple(int(c) for c in colors[track_id % len(colors)])
                label_name = CLASS_MAPPING.get(track.class_id, "Inconnu")
                state = track.current_state
                distance = float(
                    np.sqrt(
                        state["X"] ** 2 + state["Y"] ** 2 + state["Z"] ** 2
                    )
                )

                status = " (Cache)" if track.lost_frames > 0 else ""
                text = f"ID:{track_id} {label_name}{status} {distance:.1f}m"

                thickness = 1 if track.lost_frames > 0 else 2

                box_l = track.box_l
                cv2.rectangle(
                    frame1,
                    (int(box_l[0]), int(box_l[1])),
                    (int(box_l[2]), int(box_l[3])),
                    color,
                    thickness,
                )
                cv2.putText(
                    frame1,
                    text,
                    (int(box_l[0]), int(box_l[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                box_r = track.box_r
                cv2.rectangle(
                    frame2,
                    (int(box_r[0]), int(box_r[1])),
                    (int(box_r[2]), int(box_r[3])),
                    color,
                    thickness,
                )
                cv2.putText(
                    frame2,
                    text,
                    (int(box_r[0]), int(box_r[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            h1, w1 = frame1.shape[:2]
            frame2_resized = cv2.resize(frame2, (w1, h1))
            vis = np.hstack((frame1, frame2_resized))

            out.write(vis)
            cv2.imshow("Stereo View", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap1.release()
        cap2.release()
        out.release()
        cv2.destroyAllWindows()
        if topdown is not None:
            topdown.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent

    # Nom de la simulation issu de VIDEO_PATH (dossier dans Data)
    # On strip pour supprimer les espaces accidentels et on retombe
    # sur "Simulation_1" si la valeur est vide.
    simulation_name_env = os.getenv("VIDEO_PATH", "Simulation_1")
    simulation_name = simulation_name_env.strip() or "Simulation_1"

    # Chemin de sortie vidéo :
    # - si OUTPUT_VIDEO_PATH est défini et non vide, on l'utilise tel quel,
    # - sinon on crée un chemin par défaut sous runs/stereo/<simulation_name>/<simulation_name>.mp4
    output_env = os.getenv("OUTPUT_VIDEO_PATH", "").strip()
    if output_env:
        default_output = Path(output_env)
    else:
        default_output = root / "runs" / "stereo" / simulation_name / f"{simulation_name}.mp4"

    parser = argparse.ArgumentParser(
        description="Stéréovision spatio-temporelle avec YOLO + GlobalTracker"
    )
    parser.add_argument(
        "--video-left",
        type=Path,
        default=root / "Data" / simulation_name / "gauche.mp4",
        help="Vidéo caméra gauche",
    )
    parser.add_argument(
        "--video-right",
        type=Path,
        default=root / "Data" / simulation_name / "droite.mp4",
        help="Vidéo caméra droite",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", str(root / "yolov10n.pt"))),
        help="Poids YOLO",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Vidéo de sortie annotée",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=STEREO_BASELINE_DEFAULT,
        help="Baseline stéréo (mètres)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Fichier .npz de calibration stéréo (K1,D1,K2,D2,R,T,...).",
    )
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="Ne pas rectifier/undistort les frames (même si calibration fournie).",
    )
    parser.add_argument(
        "--no-topdown",
        action="store_true",
        help="Désactiver la vue de dessus (X-Z)",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=MAX_LOST_FRAMES_DEFAULT,
        help="Nombre max de frames d'occlusion avant suppression d'une piste",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.video_left.exists() or not args.video_right.exists():
        print(
            f"ATTENTION: Vidéos introuvables : {args.video_left} ou {args.video_right}"
        )
        return

    estimator = StereoDepthEstimator(
        str(args.model),
        baseline=args.baseline,
        max_lost_frames=args.max_lost_frames,
        calibration_path=args.calibration,
        rectify=not args.no_rectify,
    )
    estimator.process_videos(
        args.video_left,
        args.video_right,
        args.output,
        show_topdown=not args.no_topdown,
    )


if __name__ == "__main__":
    main()