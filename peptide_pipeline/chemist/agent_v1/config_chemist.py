from pydantic import BaseModel
from pydantic import field_validator
from typing import Optional

class RangeTarget(BaseModel):
    min: float
    max: float
    target: float
    wheight: Optional[float] = 1.0

    @field_validator("target")
    def check_target(cls, v, info):
        data = info.data
        if "min" in data and "max" in data:
            if not (data["min"] <= v <= data["max"]):
                raise ValueError("target must be within range")
        return v
    
class ChemistConfig(BaseModel):
    
    ph : float = 7.0
    length : Optional[RangeTarget] = None
    molecular_weight : Optional[RangeTarget] = None
    logp : Optional[RangeTarget] = None
    net_charge : Optional[RangeTarget] = None
    isoelectric_point : Optional[RangeTarget] = None
    hydrophobicity : Optional[RangeTarget] = None
    cathionicity : Optional[RangeTarget] = None