"""Execute README.md python snippets verbatim and assert their results.

The README ships executable example code. Each fenced ```python block whose
first content line is `# snippet: <name>` is extracted from the rendered
README and run through `exec()` against a fresh namespace. The post-exec
namespace is then asserted against the expected behaviour of the snippet.

If a test fails the README is out of sync with the public API — fix the
README itself, not the test. The test never re-implements snippet code, so
the README and tests cannot drift.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import dart_mlci

README = (Path(dart_mlci.__file__).resolve().parent.parent / "README.md").read_text()

# Match ```python ... # snippet: <name> ... ``` blocks.
_SNIPPET_RE = re.compile(
    r"```python\n(?P<body># snippet: (?P<name>[\w-]+)\n.*?)```",
    re.S,
)


def _snippets() -> dict[str, str]:
    return {m.group("name"): textwrap.dedent(m.group("body")) for m in _SNIPPET_RE.finditer(README)}


SNIPPETS = _snippets()


def _run(name: str) -> dict:
    code = SNIPPETS.get(name)
    if code is None:
        raise AssertionError(
            f"README snippet {name!r} not found. Tagged snippets present: {sorted(SNIPPETS)}"
        )
    ns: dict = {"__name__": "__readme_snippet__"}
    exec(compile(code, f"README.md::{name}", "exec"), ns)
    return ns


def test_readme_snippet_marker_detection():
    """`# snippet: marker-detection` — load chip + run YOLO on the sample image."""
    ns = _run("marker-detection")
    markers = ns["markers"]
    assert markers, "expected at least one detected marker"
    labels = {m["label"] for m in markers}
    assert labels <= {"cross", "circle"}, f"unexpected labels: {labels}"


def test_readme_snippet_full_pipeline():
    """`# snippet: full-pipeline` — detect→match→rotate→mask returns cropped image + mask."""
    ns = _run("full-pipeline")
    cropped, mask = ns["cropped"], ns["chamber_mask"]
    assert cropped.ndim == 3 and cropped.shape[-1] == 3, (
        f"unexpected cropped shape: {cropped.shape}"
    )
    assert mask.shape == cropped.shape[:2], (
        f"mask shape {mask.shape} does not match cropped {cropped.shape[:2]}"
    )
    assert cropped.size > 0 and mask.size > 0


def test_readme_snippet_list_chamber_types():
    """`# snippet: list-chamber-types` — chip JSON exposes expected chamber types."""
    ns = _run("list-chamber-types")
    assert "NormaleBox-inner" in ns["lib"].polygon_library
