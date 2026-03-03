from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib optionnel
    plt = None


@dataclass
class TopDownView:
    """
    Gestion d'une fenêtre Matplotlib montrant les entités en vue de dessus (plan X-Z).
    """

    enabled: bool = field(init=False, default=False)
    fig: Optional["plt.Figure"] = field(init=False, default=None)
    ax: Optional["plt.Axes"] = field(init=False, default=None)
    scatter: Optional["plt.Collection"] = field(init=False, default=None)
    texts: Dict[int, "plt.Text"] = field(init=False, default_factory=dict)

    x_limits: Optional[tuple[float, float]] = (-7.0, 4.0)
    z_limits: Optional[tuple[float, float]] = (0.0, 16.0)

    def __post_init__(self) -> None:
        if plt is None:
            self.enabled = False
            return

        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(num="Vue de dessus (X-Z)")
            self.ax.set_title("Coordonnées des entités (vue de dessus)")
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Z (m)")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect("equal", adjustable="box")

            if self.x_limits is not None:
                self.ax.set_xlim(*self.x_limits)
            if self.z_limits is not None:
                self.ax.set_ylim(*self.z_limits)

            self.scatter = self.ax.scatter([], [], s=40)
            self.enabled = True
        except Exception:
            self.enabled = False

    def update(self, tracks: Dict[int, "KalmanTrack"], colors: np.ndarray) -> None:
        if not self.enabled or self.fig is None or self.ax is None or self.scatter is None:
            return

        try:
            pts = []
            cols = []
            live_ids = set(tracks.keys())

            for tid in list(self.texts.keys()):
                if tid not in live_ids:
                    self.texts[tid].remove()
                    del self.texts[tid]

            for track_id, track in tracks.items():
                if (
                    track.lost_frames == 0
                    and getattr(track, "last_observed_point_3d", None) is not None
                ):
                    x = float(track.last_observed_point_3d[0])
                    z = float(track.last_observed_point_3d[2])
                else:
                    state = track.current_state
                    x = float(state["X"])
                    z = float(state["Z"])

                pts.append([x, z])
                c = colors[track_id % len(colors)] / 255.0
                cols.append(c)

                label = f"{track_id}"
                if track_id in self.texts:
                    self.texts[track_id].set_position((x, z))
                    self.texts[track_id].set_text(label)
                    self.texts[track_id].set_color(c)
                else:
                    self.texts[track_id] = self.ax.text(
                        x, z, label, fontsize=9, color=c, ha="left", va="bottom"
                    )

            if len(pts) == 0:
                self.scatter.set_offsets(np.empty((0, 2)))
            else:
                pts_arr = np.asarray(pts, dtype=float)
                self.scatter.set_offsets(pts_arr)
                self.scatter.set_color(cols)

            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            self.enabled = False

    def close(self) -> None:
        if not self.enabled or plt is None:
            return
        try:
            plt.ioff()
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:
            pass

