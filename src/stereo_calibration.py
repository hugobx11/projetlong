from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

RECTIFICATION_ALPHA = float(os.getenv("RECTIFICATION_ALPHA", "0.0"))


@dataclass(frozen=True)
class StereoCalibration:
    """
    Calibration stéréo (caméras gauche/droite) + paramètres de rectification.

    Contenu minimal attendu dans un .npz:
    - K1, D1, K2, D2 : intrinsèques + distorsion
    - R, T           : extrinsèques (gauche -> droite)
    - image_size     : (w, h)
    """

    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple[int, int]

    # Rectification (optionnel si stocké, sinon calculable)
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None

    @staticmethod
    def load_npz(path: Path) -> "StereoCalibration":
        data = np.load(str(path), allow_pickle=False)

        def must(key: str) -> np.ndarray:
            if key not in data:
                raise KeyError(f"Calibration invalide: clé manquante '{key}' dans {path}")
            return data[key]

        image_size_arr = must("image_size").astype(int).flatten()
        if image_size_arr.size != 2:
            raise ValueError("image_size doit être un tableau de 2 valeurs: (w, h)")

        calib = StereoCalibration(
            K1=must("K1"),
            D1=must("D1"),
            K2=must("K2"),
            D2=must("D2"),
            R=must("R"),
            T=must("T"),
            image_size=(int(image_size_arr[0]), int(image_size_arr[1])),
            R1=data["R1"] if "R1" in data else None,
            R2=data["R2"] if "R2" in data else None,
            P1=data["P1"] if "P1" in data else None,
            P2=data["P2"] if "P2" in data else None,
            Q=data["Q"] if "Q" in data else None,
        )
        return calib

    def with_rectification(self, alpha: float = RECTIFICATION_ALPHA) -> "StereoCalibration":
        """
        Calcule R1,R2,P1,P2,Q via stereoRectify si non présents.
        """
        if self.P1 is not None and self.P2 is not None and self.R1 is not None and self.R2 is not None:
            return self

        w, h = self.image_size
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K1,
            self.D1,
            self.K2,
            self.D2,
            (w, h),
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
        )

        return StereoCalibration(
            K1=self.K1,
            D1=self.D1,
            K2=self.K2,
            D2=self.D2,
            R=self.R,
            T=self.T,
            image_size=self.image_size,
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            Q=Q,
        )

    def build_rectification_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (map1x, map1y, map2x, map2y) pour cv2.remap.
        """
        calib = self.with_rectification()
        assert calib.R1 is not None and calib.R2 is not None and calib.P1 is not None and calib.P2 is not None

        w, h = calib.image_size
        map1x, map1y = cv2.initUndistortRectifyMap(
            calib.K1, calib.D1, calib.R1, calib.P1, (w, h), cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            calib.K2, calib.D2, calib.R2, calib.P2, (w, h), cv2.CV_32FC1
        )
        return map1x, map1y, map2x, map2y

