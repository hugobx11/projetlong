#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _iter_images(directory: Path) -> list[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts]
    return paths


def calibrate_stereo_from_chessboard(
    left_dir: Path,
    right_dir: Path,
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
) -> dict:
    left_imgs = _iter_images(left_dir)
    right_imgs = _iter_images(right_dir)
    if len(left_imgs) == 0 or len(right_imgs) == 0:
        raise ValueError("Aucune image trouvée (gauche ou droite).")

    n = min(len(left_imgs), len(right_imgs))
    left_imgs = left_imgs[:n]
    right_imgs = right_imgs[:n]

    pattern_size = (pattern_cols, pattern_rows)

    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    good = 0
    for lp, rp in zip(left_imgs, right_imgs):
        img_l = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        img_r = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            continue

        if image_size is None:
            h, w = img_l.shape[:2]
            image_size = (w, h)

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

        if not ret_l or not ret_r:
            continue

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        good += 1

    if good < 5:
        raise RuntimeError(
            f"Pas assez de paires valides avec damier détecté ({good}). Vise 10-20+."
        )

    assert image_size is not None

    # Calibrations mono
    ret_l, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, image_size, None, None)
    ret_r, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None)

    # Calibration stéréo
    flags = cv2.CALIB_FIX_INTRINSIC
    stereoret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K1,
        D1,
        K2,
        D2,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    # Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    return {
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "image_size": np.array(image_size, dtype=int),
        "mono_rms_left": np.array([ret_l], dtype=float),
        "mono_rms_right": np.array([ret_r], dtype=float),
        "stereo_rms": np.array([stereoret], dtype=float),
        "pairs_used": np.array([good], dtype=int),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration stéréo OpenCV (damier)")
    p.add_argument("--left-dir", type=Path, required=True, help="Dossier images damier gauche")
    p.add_argument("--right-dir", type=Path, required=True, help="Dossier images damier droite")
    p.add_argument("--cols", type=int, required=True, help="Nb coins intérieurs (colonnes)")
    p.add_argument("--rows", type=int, required=True, help="Nb coins intérieurs (lignes)")
    p.add_argument("--square-size-m", type=float, required=True, help="Taille d'une case (m)")
    p.add_argument("--out", type=Path, required=True, help="Fichier .npz de sortie")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    calib = calibrate_stereo_from_chessboard(
        args.left_dir, args.right_dir, args.cols, args.rows, args.square_size_m
    )
    np.savez(str(args.out), **calib)
    print(
        f"OK: calibration écrite dans {args.out} (paires utilisées: {int(calib['pairs_used'][0])}, "
        f"stereo_rms: {float(calib['stereo_rms'][0]):.4f})"
    )


if __name__ == "__main__":
    main()

