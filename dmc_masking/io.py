"""IO functionality"""

import json


def load_roi_structures(path):
    with open(path, encoding="utf-8") as input:
        roi_structures = json.load(input)

    return roi_structures
