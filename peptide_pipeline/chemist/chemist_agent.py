from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import SDWriter, rdDepictor

from peptide_pipeline.chemist.base import BaseChemist
from .config_chemist import ChemistConfig, RangeTarget
from peptide_pipeline.chemist.properties import CHEMIST_PROPERTIES

class ChemistAgent(BaseChemist):
    def __init__(self, config: ChemistConfig):
        super().__init__()
        self.config = config

    def check_validity(self, peptides: List[str]) -> List[bool]:
        """Return a list indicating whether each peptide is within the configured constraints."""
        results = []
        for peptide in peptides:
            _, in_limits = self.evaluate_peptide(peptide, self.config)
            results.append(in_limits)
        return results

    def calculate_score(self, value: float, range_target: RangeTarget) -> Tuple[float, bool]:
        """
        Calculate a score based on the distance from the target value and the limits.
        The score is higher when the value is closer to the target and decreases as it moves away from the target, especially beyond the limits.
        """
        in_limit = True
        if value < range_target.min or value > range_target.max:
            in_limit = False
            
        score = abs(value - range_target.target)
        
        return score, in_limit

    def evaluate_peptide(self, peptide: str, config: Optional[ChemistConfig] = None) -> Tuple[float, bool]:
        """
        Evaluates a single peptide against the provided chemical constraints and calculates a score for each property.
        Return the score calculated on the properties with a wheight if provided in the config and if the peptide is within all limits or not.
        """

        active_config = config or self.config

        scores = 0.0
        in_limits = True
        for name in ChemistConfig.model_fields:
            if name == "ph":
                continue

            param = getattr(active_config, name)
            if param is None:
                continue

            if name == "net_charge":
                value = CHEMIST_PROPERTIES[name](peptide, pH=active_config.ph)
            else:
                value = CHEMIST_PROPERTIES[name](peptide)

            score, in_limit = self.calculate_score(value, param)
            if param.wheight is not None:
                score *= param.wheight
            if not in_limit:
                in_limits = False
            scores += score

        return scores, in_limits

    def filter_peptides(
        self,
        peptides: List[str],
        config: Optional[ChemistConfig] = None,
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Filters a list of peptides based on chemical constraints defined in the ChemistConfig.
        If too few peptides are returned after filtering, a minimum of x% peptides will be returned based on their score (score is calculated by the target and limit distance).    
        """
        active_config = config or self.config
        peptide_ranking = {}
        for peptide in peptides:
            scores, in_limits = self.evaluate_peptide(peptide, active_config)
            peptide_ranking[peptide] = (scores, in_limits)
        
        # First filter peptides that are within limits
        filtered_peptides = [pep for pep, (scores, in_limits) in peptide_ranking.items() if in_limits]

        if len(filtered_peptides) < threshold * len(peptides):
            # If too few peptides are within limits, include those closest to targets first.
            sorted_peptides = sorted(peptide_ranking.items(), key=lambda item: item[1][0])
            additional_peptides = [pep for pep, (scores, in_limits) in sorted_peptides if pep not in filtered_peptides]
            filtered_peptides.extend(additional_peptides[:int(threshold * len(peptides)) - len(filtered_peptides)])

        return filtered_peptides
    
    def calculate_properties(self, peptide: str, config: Optional[ChemistConfig] = None) -> Dict[str, float]:
        """
        Calculates the properties of a peptide based on the provided ChemistConfig.
        Returns a dictionary with property names as keys and their corresponding values as values.
        """
        active_config = config or self.config
        properties = {}
        for name in ChemistConfig.model_fields:
            if name == "ph":
                continue

            if getattr(active_config, name) is None:
                continue

            if name == "net_charge":
                properties[name] = CHEMIST_PROPERTIES[name](peptide, pH=active_config.ph)
            else:
                properties[name] = CHEMIST_PROPERTIES[name](peptide)
        
        return properties

    def analyze_peptide(self, peptide: str) -> Dict[str, object]:
        score, in_limits = self.evaluate_peptide(peptide)
        properties = self.calculate_properties(peptide)
        return {
            "sequence": peptide,
            "in_limits": in_limits,
            "score": score,
            "properties": properties,
        }

    def create_sdf_file(self, peptides: List[str], path: str):
        """
        Computes a .sdf file for each peptide in the list
        .sdf includes 2DCoords and properties computed by the calculate_properties() function
        """
        with SDWriter(path) as writer:
            for seq in peptides:
                props = self.calculate_properties(seq)
                mol = Chem.MolFromSequence(seq)
                if mol is None:
                    self.logger.warning(f"Peptide not constructible from sequence {seq}")
                    continue
                rdDepictor.Compute2DCoords(mol)
                
                for key, value in props.items():
                    mol.SetProp(key, str(value))
                
                writer.write(mol)







