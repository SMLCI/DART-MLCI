import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns


@dataclass
class AffineTransformResult:
    """Result of computing an affine transform between two maps."""

    transform: Callable[[np.ndarray], np.ndarray]
    residuals: np.ndarray  # Per-point residual distances
    rmse: float  # Root mean square error
    max_error: float  # Maximum residual


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

    @staticmethod
    def from_dict_list(entries: list[dict]) -> "Map":
        """Create a Map from a list of dicts (e.g., from chip config blueprint_map).

        Each dict must have 'roi_id', 'x', and 'y' keys.

        Args:
            entries: List of dicts with roi_id, x, y fields

        Returns:
            Map object with positions from the entries
        """
        roi_positions = []
        for entry in entries:
            roi_id = f"{int(entry['roi_id']):04d}"
            roi_positions.append(
                RoIPosition(roi_id=roi_id, position=np.array([entry["x"], entry["y"]], dtype=float))
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

    def to_csv(self, output_path: Path, z_positions: dict[str, float] | None = None) -> None:
        """Write the map to a CSV file with roi_id, x, y, z columns.

        Args:
            output_path: Destination CSV path.
            z_positions: Optional mapping from roi_id to z coordinate. RoIs not in
                the mapping receive the mean of the provided z values. If None or
                empty, z is set to 0.0 for every row.
        """
        df = self.to_df()
        if z_positions:
            avg_z = float(np.mean(list(z_positions.values())))
            df["z"] = df["roi_id"].map(lambda rid: z_positions.get(rid, avg_z))
        else:
            df["z"] = 0.0
        df.to_csv(output_path, index=False)

    def compute_affine_transform(self, target: "Map") -> AffineTransformResult:
        """Compute an affine transform from this map to the target map.

        Uses least squares fitting to handle any number of corresponding points (>=3).
        Returns the transform function along with error metrics.

        Args:
            target: The target map with corresponding RoI positions

        Returns:
            AffineTransformResult containing the transform function and error metrics

        Raises:
            AssertionError: If fewer than 3 points are provided
            KeyError: If a target RoI ID is not found in this map
        """
        ids = list(target.roi_positions.keys())
        n_points = len(ids)

        assert n_points >= 3, "Need at least 3 points for affine transform"

        # get the corresponding points from the blueprint positions
        blueprint_points = np.stack([self.roi_positions[rid].position for rid in ids])

        # convert measured points to np.array
        measured_points = np.stack([target.roi_positions[rid].position for rid in ids])

        # Build design matrix: [x, y, 1] for each point
        design = np.hstack((blueprint_points, np.ones((n_points, 1))))

        # Solve using least squares: design @ Ab = measured_points
        Ab, _residuals, _rank, _s = np.linalg.lstsq(design, measured_points, rcond=None)

        # Create transform function (handles both single points and batches)
        def transform(x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                return (x @ Ab[:2, :] + Ab[2:3, :])[0]
            return x @ Ab[:2, :] + Ab[2:3, :]

        # Compute per-point residuals
        transformed = transform(blueprint_points)
        point_residuals = np.linalg.norm(transformed - measured_points, axis=1)

        return AffineTransformResult(
            transform=transform,
            residuals=point_residuals,
            rmse=float(np.sqrt(np.mean(point_residuals**2))),
            max_error=float(np.max(point_residuals)),
        )

    def apply_transform(
        self, t: AffineTransformResult | Callable[[np.ndarray], np.ndarray]
    ) -> "Map":
        """Apply a transform to all RoI positions in this map.

        Args:
            t: Either an AffineTransformResult or a callable transform function

        Returns:
            A new Map with transformed positions
        """
        # Handle both AffineTransformResult and raw callable
        transform_fn = t.transform if isinstance(t, AffineTransformResult) else t

        # Batch transform all positions at once
        roi_ids = list(self.roi_positions.keys())
        positions = np.stack([self.roi_positions[rid].position for rid in roi_ids])
        transformed = transform_fn(positions)

        # Create new RoI positions from transformed coordinates
        new_roi_positions = [RoIPosition(rid, transformed[i]) for i, rid in enumerate(roi_ids)]

        return Map(new_roi_positions)
