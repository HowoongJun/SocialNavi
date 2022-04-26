
from dataclasses import dataclass

@dataclass
class CDetectedData():
    data_class: int = None
    confidence: float = None
    xmin: float = None
    ymin: float = None
    xmax: float = None
    ymax: float = None
    obj_id: int = None
    depth: float = None