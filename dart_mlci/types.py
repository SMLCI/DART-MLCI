"""Domain types for pipeline results, timings, and errors."""

from __future__ import annotations

from dataclasses import dataclass, field


class PipelineError(Exception):
    """Pipeline step failure with step name tracking.

    Attributes:
        step: Name of the pipeline step that failed.
        message: Human-readable error description.
    """

    def __init__(self, step: str, message: str):
        self.step = step
        self.message = message
        super().__init__(f"{step}: {message}")


@dataclass
class PipelineTimings:
    """Timing metrics for pipeline steps (in seconds)."""

    detection: float = 0.0
    matching: float = 0.0
    rotation: float = 0.0
    registration: float = 0.0
    masking: float = 0.0
    segmentation: float = 0.0

    @property
    def total(self) -> float:
        """Sum of all step times."""
        return (
            self.detection
            + self.matching
            + self.rotation
            + self.registration
            + self.masking
            + self.segmentation
        )

    def as_dict(self) -> dict[str, float]:
        """Return dict with t_-prefixed keys for CSV export."""
        return {
            "t_detection": self.detection,
            "t_matching": self.matching,
            "t_rotation": self.rotation,
            "t_registration": self.registration,
            "t_masking": self.masking,
            "t_segmentation": self.segmentation,
            "t_total": self.total,
        }


@dataclass
class FrameResult:
    """Result for processing a single frame."""

    success: bool = False
    failed_step: str | None = None
    error_message: str | None = None
    n_cells: int = 0
    timings: PipelineTimings = field(default_factory=PipelineTimings)


@dataclass
class StackResult:
    """Result for processing a TIFF stack."""

    folder: str
    file_name: str
    chamber_type: str
    n_frames: int = 0
    n_success: int = 0
    n_cells_total: int = 0
    success: bool = False
    error: str = ""
