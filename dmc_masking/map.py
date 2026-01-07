import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns


class RoIPosition:
    """Region of Interest (e.g. growth chamber) on a map."""

    def __init__(self, roi_id: str, position: npt.NDArray[np.float64]):
        self.id = roi_id
        self.position = position

        if self.position.shape not in ((2,), (3,)):
            raise ValueError("Invalid dimensions for a RoI position")

    def __sub__(self, other) -> np.ndarray:
        if self.position.shape != other.position.shape:
            raise ValueError("RoIs on the map have different position dimensions!")
        return self.position - other.position

    def __repr__(self) -> str:
        return f"MapRoi({self.id=}, {self.position=})"


class Map:
    """Map containing regions of interest (RoIs)."""

    def __init__(self, roi_poisitions: list[RoIPosition]):
        self.roi_positions = {r.id: r for r in roi_poisitions}

        if len(self.roi_positions) == 0:
            logging.warning("Creating empty map")

    def rel_movement_from_to(self, id_from: str | np.ndarray, id_to: str) -> np.ndarray:
        """Compute the relative movement between two RoIs.

        Args:
            id_from (str): movement start RoI
            id_to (str): movement target RoI

        Returns:
            np.ndarray: movement vector to move from id_from to id_do
        """

        if isinstance(id_from, str):
            id_from = self.roi_positions[id_from].position

        if isinstance(id_to, str):
            id_to = self.roi_positions[id_to].position

        # compute the relative movement needed
        return id_to - id_from

    def __getitem__(self, index: str) -> RoIPosition:
        return self.roi_positions[index]

    # @match_typing
    def distance(self, id_from: str, id_to: str) -> float:
        """Computes the euclidean distance between two RoIs on the map.

        Args:
            id_from (str): start RoI
            id_to (str): target RoI

        Returns:
            float: Euclidean distance between id_from and id_to RoIs
        """
        return np.linalg.norm(self.roi_positions[id_from] - self.roi_positions[id_to])

    @staticmethod
    def from_csv(csv_map_file: Path) -> "Map":
        df_map = pd.read_csv(csv_map_file)
        df_map["roi_id"] = df_map["roi_id"].apply(lambda rid: f"{rid:04d}")

        roi_positions = []

        for _, row in df_map.iterrows():
            roi_positions.append(
                RoIPosition(roi_id=row["roi_id"], position=np.array([row["x"], row["y"]]))
            )

        return Map(roi_positions)

    def plot(self, ax=None):
        df = pd.DataFrame(
            [
                {
                    "id": roi_id,
                    "x": r.position[0],
                    "y": r.position[1],
                }
                for roi_id, r in self.roi_positions.items()
            ]
        )

        # plot all the chamber positions
        sns.scatterplot(data=df, x="x", y="y", ax=ax)

    def to_df(self):
        df = pd.DataFrame(
            [
                {
                    "roi_id": roi_id,
                    "x": r.position[0],
                    "y": r.position[1],
                }
                for roi_id, r in self.roi_positions.items()
            ]
        )

        return df

    def compute_affine_transform(self, target: "Map"):
        ids = target.roi_positions.keys()

        assert len(ids) == 3, "We can only do this for triplets right now"

        # get the corresponding points from the blueprint positions
        blueprint_points = np.stack([self.roi_positions[rid].position for rid in ids])

        # convert measured points to np.array
        measured_points = np.stack([target.roi_positions[rid].position for rid in ids])

        def create_affine_function(pointsA: np.ndarray, pointsB: np.ndarray):
            _pointsB = np.hstack((pointsB, np.ones((3, 1))))
            Ab = np.linalg.solve(_pointsB, pointsA)
            return lambda x: Ab[:2, :].T.dot(x.T).T + Ab[2:3, :]

        # get affine transformation
        f = create_affine_function(measured_points, blueprint_points)

        # return the affine transform
        return f

    def apply_transform(self, t) -> "Map":
        new_roi_positions = []

        # apply the transform to all RoI positions
        for roi_id, rp in self.roi_positions.items():
            new_roi_positions.append(RoIPosition(roi_id, t(rp.position[None])[0]))

        # return the new map
        return Map(new_roi_positions)
