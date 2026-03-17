from typing import Dict, List, Optional, Tuple
from numpy import exp
from rdkit import Chem
from rdkit.Chem import SDWriter, rdDepictor

from peptide_pipeline.chemist.base import BaseChemist
from .config_chemist import ChemistConfig, RangeTarget
from .properties import PROPERTY_REGISTRY


class ChemistAgent(BaseChemist):
    def __init__(self, config: ChemistConfig):
        super().__init__(config)
        self.config = config
    
    def _calculate_properties(self, peptide: str) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """
        Calculates the properties of a peptide based on the provided ChemistConfig.
        Returns a dictionary with property names as keys and their corresponding values as values.
        """
        properties = {}
        distance_from_target = {}
        in_limits = True
        
        for prop_name, prop_def in PROPERTY_REGISTRY.items():
            # Skip if not configured
            if getattr(self.config, prop_name) is None:
                continue
            
            # Calculate property value
            if prop_def.requires_ph:
                properties[prop_name] = prop_def.function(peptide, pH=self.config.ph)
            else:
                properties[prop_name] = prop_def.function(peptide)
            
            # Check limits
            config_constraint = getattr(self.config, prop_name)
            if config_constraint.min is not None and properties[prop_name] < config_constraint.min:
                in_limits = False
            if config_constraint.max is not None and properties[prop_name] > config_constraint.max:
                in_limits = False
            
            # Calculate distance from target
            if config_constraint.target is not None:
                distance_from_target[prop_name] = abs(properties[prop_name] - config_constraint.target)
            else:
                self.logger.warning(f"No target defined for property '{prop_name}' in ChemistConfig. Distance from target will not be calculated for this property.")

        return properties, distance_from_target, in_limits

    def _analyze_peptide(self, peptide: str) -> Dict[str, object]:
        properties, distance_from_target, in_limits = self._calculate_properties(peptide)
        self.logger.debug(f"Analyzed peptide: {peptide}, Properties: {properties}, Distance from target: {distance_from_target}, In limits: {in_limits}")
        return {
            "sequence": peptide,
            "properties": properties,
            "distance_from_target": distance_from_target,
            "in_limits": in_limits
        }
    
    def evaluate_peptides(self, peptides: List[str]) -> List[Dict[str, object]]:
        """
        Evaluates a list of peptides against the provided chemical constraints and calculates scores for each property.
        Returns a list of dictionaries, each containing property names as keys and their corresponding scores as values for each peptide.
        Also includes a boolean indicating if each peptide is within all limits or not.
        """
        if not peptides:
            return []

        analyzed_peptides = {peptide: self._analyze_peptide(peptide) for peptide in peptides}

        # list of properties "valid" for the config (not None) and not ph
        valid_properties = [
            prop_name
            for prop_name in PROPERTY_REGISTRY.keys()
            if getattr(self.config, prop_name) is not None
        ]

        # Track overall weighted scores to avoid separate iterations
        overall_scores = {peptide: 0.0 for peptide in analyzed_peptides}

        # Calculate z-scores for each property and accumulate weighted scores
        for name in valid_properties:
            # Get pairs of peptide with distance from target for property "name"
            pairs = [
                (peptide, result["distance_from_target"][name])
                for peptide, result in analyzed_peptides.items()
                if name in result["distance_from_target"]
            ]
            if not pairs:
                continue

            # Calculate min and max distances for this property
            values = [value for _, value in pairs]
            min_val = min(values)
            max_val = max(values)

            # Get weight once per property for efficiency
            config_property = getattr(self.config, name)
            weight = config_property.weight if config_property is not None and config_property.weight is not None else 1.0

            # Calculate normalized distance and accumulate weighted score
            for peptide, value in pairs:
                z_score = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
                
                # Initialize norm_score on first use
                if "norm_score" not in analyzed_peptides[peptide]:
                    analyzed_peptides[peptide]["norm_score"] = {}
                analyzed_peptides[peptide]["norm_score"][name] = float(z_score)
                
                # Accumulate weighted score
                overall_scores[peptide] += z_score * weight

        # Normalize overall scores and build final output in a single pass
        overall_values = list(overall_scores.values())
        if overall_values:
            min_overall = min(overall_values)
            max_overall = max(overall_values)
        else:
            min_overall = max_overall = 0.0

        scored_peptides = []
        for peptide in analyzed_peptides.keys():
            result = analyzed_peptides[peptide]
            # Normalize score to [0, 1] range where 1 is best (lowest normalized score)
            if max_overall == min_overall:
                score = 1.0
            else:
                overall_score = overall_scores[peptide]
                score = 1.0 - ((overall_score - min_overall) / (max_overall - min_overall))
            
            scored_peptides.append(
                {
                    "sequence": peptide,
                    "properties": result["properties"],
                    "score": score,
                    "in_limits": result["in_limits"],
                }
            )
        return scored_peptides
    
    def get_top_filtered_peptides(self, peptides: List[str], topK: int) -> List[str]:
        peptide_evaluated = self.evaluate_peptides(peptides)
        # Sort by: in_limits peptides first (True first), then by score (descending)
        sorted_peptides = sorted(peptide_evaluated, key=lambda x: (not x["in_limits"], -x["score"]))
        
        # Log if not enough in-limit peptides
        in_limit_count = sum(1 for p in sorted_peptides if p["in_limits"])
        if in_limit_count < topK:
            self.logger.info(f"Only {in_limit_count} peptides are within limits, returning {topK - in_limit_count} out of limits ranked by score.")
        
        return sorted_peptides[:topK]





