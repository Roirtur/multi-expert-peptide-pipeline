import peptides as pp
from rdkit import Chem, logger
from rdkit.Chem import Descriptors

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


CHEMIST_PROPERTIES = {
    "length": compute_length,
    "molecular_weight": compute_molecular_weight,
    "logp": compute_logp,
    "net_charge": compute_net_charge,
    "isoelectric_point": compute_isoelectric_point,
    "hydrophobicity": compute_hydrophobicity,
    "cathionicity": compute_cathionicity
}