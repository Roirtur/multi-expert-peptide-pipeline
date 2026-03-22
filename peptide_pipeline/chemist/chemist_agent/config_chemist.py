from pydantic import BaseModel
from pydantic import field_validator, model_validator
from typing import Optional

class RangeTarget(BaseModel):
    min: float
    max: float
    target: float
    weight: Optional[float] = 1.0

    @model_validator(mode='after')
    def check_min_max(self):
        if self.min > self.max:
            raise ValueError("min must be less than or equal to max")
        return self

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
