import os
from dotenv import load_dotenv
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


load_dotenv()

KALMAN_PROCESS_NOISE = float(os.getenv("KALMAN_PROCESS_NOISE", "0.05"))
KALMAN_MEASUREMENT_NOISE = float(os.getenv("KALMAN_MEASUREMENT_NOISE", "0.5"))

MAX_LOST_FRAMES = int(os.getenv("MAX_LOST_FRAMES", "5"))

MAX_Y_DIFF = float(os.getenv("MAX_Y_DIFF", "30"))
MAX_SIZE_DIFF = float(os.getenv("MAX_SIZE_DIFF", "50"))

MAX_TEMP_COST = float(os.getenv("MAX_TEMP_COST", "1000"))
MAX_PIXEL_DIST = float(os.getenv("MAX_PIXEL_DIST", "200"))
MAX_DEPTH_DIST = float(os.getenv("MAX_DEPTH_DIST", "15"))


class KalmanTrack:
    """
    Représente une piste (track) suivie dans le temps.
    Suit la position 3D (X, Y, Z) et sa projection image (cx, cy) avec un Kalman.
    """

    def __init__(self, track_id: int, obs: dict):
        self.id = track_id
        self.class_id = obs["class"]
        self.lost_frames = 0
        self.box_l = obs["box_l"]
        self.box_r = obs["box_r"]

        # Dernière observation 3D brute (triangulation)
        self.last_observed_point_3d = np.array(
            [obs["X"], obs["Y"], obs["Z"]], dtype=float
        )

        # Filtre de Kalman : 10 variables d'état (X,Y,Z,cx,cy,vX,vY,vZ,vcx,vcy), 5 mesures
        self.kf = cv2.KalmanFilter(10, 5)

        # Transition (modèle vitesse constante)
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        for i in range(5):
            self.kf.transitionMatrix[i, i + 5] = 1.0

        # Mesure (on observe directement X,Y,Z,cx,cy)
        self.kf.measurementMatrix = np.zeros((5, 10), dtype=np.float32)
        for i in range(5):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = (
            np.eye(5, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        )
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)

        state = np.array(
            [obs["X"], obs["Y"], obs["Z"], obs["cx"], obs["cy"], 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        self.kf.statePost = state.reshape(-1, 1)
        self.kf.statePre = state.reshape(-1, 1)

        self.current_state = {
            "X": obs["X"],
            "Y": obs["Y"],
            "Z": obs["Z"],
            "cx": obs["cx"],
            "cy": obs["cy"],
        }

    def predict(self) -> dict:
        old_cx, old_cy = self.current_state["cx"], self.current_state["cy"]
        pred = self.kf.predict()

        self.current_state["X"] = float(pred[0, 0])
        self.current_state["Y"] = float(pred[1, 0])
        self.current_state["Z"] = float(pred[2, 0])
        self.current_state["cx"] = float(pred[3, 0])
        self.current_state["cy"] = float(pred[4, 0])

        dx = self.current_state["cx"] - old_cx
        dy = self.current_state["cy"] - old_cy

        self.box_l = [
            self.box_l[0] + dx,
            self.box_l[1] + dy,
            self.box_l[2] + dx,
            self.box_l[3] + dy,
        ]
        self.box_r = [
            self.box_r[0] + dx,
            self.box_r[1] + dy,
            self.box_r[2] + dx,
            self.box_r[3] + dy,
        ]

        return self.current_state

    def update(self, obs: dict) -> None:
        meas = np.array(
            [obs["X"], obs["Y"], obs["Z"], obs["cx"], obs["cy"]], dtype=np.float32
        )
        self.kf.correct(meas.reshape(-1, 1))
        self.box_l = obs["box_l"]
        self.box_r = obs["box_r"]
        self.lost_frames = 0

        post = self.kf.statePost
        self.current_state["X"] = float(post[0, 0])
        self.current_state["Y"] = float(post[1, 0])
        self.current_state["Z"] = float(post[2, 0])
        self.current_state["cx"] = float(post[3, 0])
        self.current_state["cy"] = float(post[4, 0])

        self.last_observed_point_3d = np.array(
            [obs["X"], obs["Y"], obs["Z"]], dtype=float
        )


class GlobalTracker:
    """
    Gère l'ensemble des pistes globales (multi-objets, multi-caméras, multi-frames).
    """

    def __init__(
        self, P1: np.ndarray, P2: np.ndarray, max_lost_frames: int = MAX_LOST_FRAMES
    ):
        self.P1 = P1
        self.P2 = P2
        self.tracks: dict[int, KalmanTrack] = {}
        self.next_id = 1
        self.max_lost_frames = max_lost_frames

    def associate_stereo(
        self,
        dets_left: list[dict],
        dets_right: list[dict],
        max_y_diff: float = MAX_Y_DIFF,
        max_size_diff: float = MAX_SIZE_DIFF,
    ) -> list[tuple[dict, dict]]:
        if len(dets_left) == 0 or len(dets_right) == 0:
            return []

        max_cost = 1e5
        cost_matrix = np.full((len(dets_left), len(dets_right)), max_cost)

        for i, det_l in enumerate(dets_left):
            for j, det_r in enumerate(dets_right):
                if det_l["class"] != det_r["class"]:
                    continue

                xl_c = (det_l["box"][0] + det_l["box"][2]) / 2
                yl_c = (det_l["box"][1] + det_l["box"][3]) / 2
                hl = det_l["box"][3] - det_l["box"][1]
                wl = det_l["box"][2] - det_l["box"][0]

                xr_c = (det_r["box"][0] + det_r["box"][2]) / 2
                yr_c = (det_r["box"][1] + det_r["box"][3]) / 2
                hr = det_r["box"][3] - det_r["box"][1]
                wr = det_r["box"][2] - det_r["box"][0]

                if xr_c > xl_c:
                    continue

                y_diff = abs(yl_c - yr_c)
#                if y_diff > max_y_diff:
#                    continue

                size_diff = abs(hl - hr) + abs(wl - wr)
                if size_diff > max_size_diff:
                    continue

                cost_matrix[i, j] = y_diff + (size_diff * 0.5)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: list[tuple[dict, dict]] = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] != np.inf:
                matched_pairs.append((dets_left[r], dets_right[c]))

        return matched_pairs

    def estimate_3d_position(self, det_l: dict, det_r: dict) -> np.ndarray:
        def get_bottom_center(box: np.ndarray) -> np.ndarray:
            return np.array([[(box[0] + box[2]) / 2], [box[3]]])

        pt1_f = get_bottom_center(det_l["box"]).astype(float)
        pt2_f = get_bottom_center(det_r["box"]).astype(float)

        point_4d = cv2.triangulatePoints(self.P1, self.P2, pt1_f, pt2_f)
        point_3d = point_4d[:3] / point_4d[3]
        return point_3d.flatten()

    def update_tracks(self, matched_pairs: list[tuple[dict, dict]]) -> None:
        current_frame_observations: list[dict] = []
        for det_l, det_r in matched_pairs:
            X, Y, Z = self.estimate_3d_position(det_l, det_r)
            cx = (det_l["box"][0] + det_l["box"][2]) / 2
            cy = (det_l["box"][1] + det_l["box"][3]) / 2
            current_frame_observations.append(
                {
                    "class": det_l["class"],
                    "X": X,
                    "Y": Y,
                    "Z": Z,
                    "cx": cx,
                    "cy": cy,
                    "box_l": det_l["box"],
                    "box_r": det_r["box"],
                }
            )

        track_ids = list(self.tracks.keys())
        predicted_states = {tid: self.tracks[tid].predict() for tid in track_ids}

        max_temp_cost = MAX_TEMP_COST
        cost_matrix = np.full(
            (len(track_ids), len(current_frame_observations)), max_temp_cost
        )

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            pred = predicted_states[tid]
            for j, obs in enumerate(current_frame_observations):
                if track.class_id != obs["class"]:
                    continue

                dist_pixel = np.sqrt(
                    (pred["cx"] - obs["cx"]) ** 2 + (pred["cy"] - obs["cy"]) ** 2
                )
                dist_z = abs(pred["Z"] - obs["Z"])

                if dist_pixel < MAX_PIXEL_DIST and dist_z < MAX_DEPTH_DIST:
                    cost_matrix[i, j] = dist_pixel

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_obs: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_temp_cost:
                tid = track_ids[r]
                obs = current_frame_observations[c]
                self.tracks[tid].update(obs)
                assigned_obs.add(c)

        for r in range(len(track_ids)):
            if (
                r not in row_ind
                or cost_matrix[r, col_ind[list(row_ind).index(r)]] == max_temp_cost
            ):
                self.tracks[track_ids[r]].lost_frames += 1

        for j, obs in enumerate(current_frame_observations):
            if j not in assigned_obs:
                self.tracks[self.next_id] = KalmanTrack(self.next_id, obs)
                self.next_id += 1

        self.tracks = {
            k: v for k, v in self.tracks.items() if v.lost_frames < self.max_lost_frames
        }

