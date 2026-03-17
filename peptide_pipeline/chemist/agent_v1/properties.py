import peptides as pp
from rdkit import Chem, logger
from rdkit.Chem import Descriptors
from dataclasses import dataclass
from typing import Callable, Optional

def compute_length(peptide: str) -> int:
    return len(peptide)

def compute_molecular_weight(peptide: str) -> float:
    return pp.Peptide(peptide).molecular_weight()

def compute_logp(peptide: str) -> float:
    mol = Chem.MolFromSequence(peptide)
    if mol is None:
        logger.error(f"Peptide not constructible from sequence {peptide}, RDKit could not build the molecule.")
        return -1.0         
    return Descriptors.MolLogP(mol)

def compute_net_charge(peptide: str, pH: float = 7.0) -> float:
    return pp.Peptide(peptide).charge(pH=pH)

def compute_isoelectric_point(peptide: str) -> float:
    return pp.Peptide(peptide).isoelectric_point()

def compute_hydrophobicity(peptide: str) -> float:
    return pp.Peptide(peptide).hydrophobicity()

def compute_cathionicity(peptide: str) -> float:
    return pp.Peptide(peptide).cathionicity()


@dataclass
class PropertyDefinition:
    """Defines a chemical property with its calculator and metadata"""
    name: str
    function: Callable
    description: str
    requires_ph: bool = False
    default_min: Optional[float] = None
    default_max: Optional[float] = None
    default_target: Optional[float] = None


# Registry of all available properties
PROPERTY_REGISTRY = {
    "length": PropertyDefinition(
        name="length",
        function=compute_length,
        description="Peptide chain length (number of amino acids)",
        requires_ph=False,
        default_min=1,
        default_max=50,
        default_target=10,
    ),
    "molecular_weight": PropertyDefinition(
        name="molecular_weight",
        function=compute_molecular_weight,
        description="Molecular weight in Da",
        requires_ph=False,
        default_min=100,
        default_max=2000,
        default_target=500,
    ),
    "logp": PropertyDefinition(
        name="logp",
        function=compute_logp,
        description="Lipophilicity (LogP)",
        requires_ph=False,
        default_min=-5,
        default_max=5,
        default_target=0,
    ),
    "net_charge": PropertyDefinition(
        name="net_charge",
        function=compute_net_charge,
        description="Net charge at given pH",
        requires_ph=True,
        default_min=-10,
        default_max=10,
        default_target=0,
    ),
    "isoelectric_point": PropertyDefinition(
        name="isoelectric_point",
        function=compute_isoelectric_point,
        description="Isoelectric point (pI)",
        requires_ph=False,
        default_min=3,
        default_max=12,
        default_target=7,
    ),
    "hydrophobicity": PropertyDefinition(
        name="hydrophobicity",
        function=compute_hydrophobicity,
        description="Hydrophobicity index",
        requires_ph=False,
        default_min=-5,
        default_max=5,
        default_target=0,
    ),
    "cathionicity": PropertyDefinition(
        name="cathionicity",
        function=compute_cathionicity,
        description="Cathionicity (cationic character)",
        requires_ph=False,
        default_min=0,
        default_max=1,
        default_target=0.5,
    ),
}