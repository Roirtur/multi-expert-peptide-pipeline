# Chemist Agent Module - Technical Documentation

## Overview

The **Chemist module** is responsible for evaluating and filtering peptide sequences based on chemical properties and constraints. It acts as the chemical validation and evaluation layer within the multi-expert peptide pipeline.

### Role
- **Validate** peptide chemical feasibility based on defined constraints
- **Evaluate** peptide properties (molecular weight, charge, hydrophobicity, etc.)
- **Filter** candidates based on chemical criteria
- **Score** peptides according to proximity to target property values
- **Provide structured output** for downstream pipeline components (scoring, biologist filtering, peptide property values)
  
### Initialisation input and output:

Input and Output :

- **Input**: `List[str]` of peptide sequences to evaluate (from generator) 

- **Output**: `List[Dict[str, Any]]` of topK evaluated peptide properties and scores sorted by score and filtered by valdity flags (all properties of the sequence in limits). 
- Structured output (e.g., `Chemist_output_payload`) containing:
  - `sequence`: The peptide sequence
  - `properties`: Dict of computed property values (e.g., molecular weight, charge)
  - `scores`: Aggregate score or per-property scores
  - `in_limits`: Boolean indicating if within all constraints
  
### Module Location
```
peptide_pipeline/chemist/
├── __init__.py
├── base.py              # Base agent class and output payload
├── agent_v1/            # v1 implementation
│   ├── chemist_agent.py # ChemistAgent implementation for v1
│   ├── config_chemist.py # Configuration model for agent_v1
│   ├──  properties.py   # Property calculation functions for agent_v1 and link to Configuration model
```

---

## Base Agent Architecture

### **BaseChemist** (Abstract Base Class)

Defines the contract that all chemist implementations must fulfill. It provides:

#### Class Constants
```python
basic_aa = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'}
```
Standard set of proteinogenic amino acids for validation.

#### Logger
```python
logger = logging.getLogger("peptide_pipeline.chemist")
```
Module-level logger for debugging and runtime monitoring.

#### Constructor
```python
def __init__(self, Config: BaseModel):
    self.config = Config
```

#### Sequence validation method
```python 
validate_sequence(sequence: str)
```
: Method to check if a peptide sequence contains only valid amino acids. 


## Abstract Methods (Must be Implemented)

##### 1. **get_top_filtered_peptides()**
```python
@abstractmethod
def get_top_filtered_peptides(self, peptides: List[str], topK: int) -> List[str]:
```

**Responsibility:**
- Filter peptides according to chemical constraints
- Rank and return top K candidates

**Parameters:**
- `peptides` (List[str]): List of peptide sequences to filter
- `topK` (int): Number of top candidates to return 

**Returns:**
- List[Dict[str, Any]]: List of evaluation results where each dictionary contains:
  - `sequence`: The peptide sequence
  - `properties`: Dict of computed property values
  - `scores`: Aggregate score or per-property scores
  - `in_limits`: Boolean indicating if within all constraints

**Expected Behavior:**
- Applies all configured constraints (min, max, validity checks)
- Ranks remaining peptides by proximity to target values
- Returns at most K peptides

---

##### 2. **evaluate_peptides()**
```python
@abstractmethod
def evaluate_peptides(self, peptides: List[str]) -> List[Dict[str, Any]]:
```

**Responsibility:**
- Compute all configured chemical properties for each peptide
- Calculate peptide scores based on distance from target values

**Parameters:**
- `peptides` (List[str]): List of peptide sequences to evaluate

**Returns:**
- List[Dict[str, Any]]: List of evaluation results where each dictionary contains:
  - `sequence`: The peptide sequence
  - `properties`: Dict of computed property values
  - `distance_from_target`: Dict of how far each property is from target
  - `in_limits`: Boolean indicating if within all constraints
  - `scores`: Aggregate score or per-property scores

**Expected Behavior:**
- Never raises exceptions for invalid peptides; logs and handles gracefully
- Calculates all configured property targets
- Returns complete evaluation for each peptide regardless of validity
- Logs warnings for missing or failed property calculations

---


## Integration Guide

### How to Use the Chemist Agent V1 implementation

#### Step 1: Create a Configuration

```python
from pydantic import BaseModel

class YourChemistConfig(BaseModel):
    # Define chemical constraints
    ph: float = 7.0
    molecular_weight: RangeTarget = RangeTarget(min=500, max=2000, target=1000, weight=1.5)
    net_charge: RangeTarget = RangeTarget(min=-5, max=5, target=0, weight=1.0)
    # ... other properties
```

#### Step 2: Instantiate the Agent
```python
from peptide_pipeline.chemist import ChemistAgent

config = YourChemistConfig(...)
chemist = ChemistAgent(config)
```

#### Step 3: Evaluate Peptides
```python
peptides = ["ACDE", "MVLSE", "GEWQL"]

# Option A: Evaluate all with detailed results
results = chemist.evaluate_peptides(peptides)
for result in results:
    print(f"Peptide: {result['sequence']}")
    print(f"Properties: {result['properties']}")
    print(f"Valid: {result['in_limits']}")

# Option B: Get top K filtered peptides
top_peptides = chemist.get_top_filtered_peptides(peptides, topK=2)
```

### Integration Points

The Chemist agent fits into the pipeline as:
```
[Data Input] → [Generator] → [CHEMIST] → [Biologist] → [Output/Storage]
```

**Inputs Received From:**
- Generator: List of peptide sequences to evaluate

**Outputs Sent To:**
- Biologist: Evaluate on a biological level the sequences filtered
- Orchestrator: Structured evaluation results and compute the scores for the peptides
- TUI: Property visualizations and rankings

---

## Implementations

### agent_v1: ChemistAgent

The current implementation (`agent_v1.chemist_agent.ChemistAgent`) includes:

**Configurable Properties:**
1. **Length**: Number of amino acids (default target: 10)
2. **Molecular Weight**: Mass in Daltons (default target: 500 Da)
3. **LogP**: Lipophilicity/hydrophobicity (default target: 0)
4. **Net Charge**: Electric charge at specified pH (default target: 0) __[Needs pH for calculation]__
5. **Isoelectric Point**: pH at zero charge (default target: 7)
6. **Hydrophobicity**: Hydrophobic index (default target: 0)
7. **Cathionicity**: Positive charge density (default target: varies)

Properties and calculations can be added in `properties.py` and linked to the configuration model in `config_chemist.py` for future enhancements.

**Key Methods:**
- `_calculate_properties()`: Computes all configured properties for a single peptide
- `_analyze_peptide()`: Wraps calculation with validation and distance computation
- `evaluate_peptides()`: Computes z-scores and weighted scoring across the peptide set

**Scoring Strategy:**
- Normalizes property distances using min/max scaling score
- Applies optional weights per property
- Excludes out-of-range peptides from top K
- Returns in range peptides sorted by score
