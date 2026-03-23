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
            self.logger.debug(f"Calculating property '{prop_name}' for peptide '{peptide}' using function '{prop_def.function.__name__}' with pH={self.config.ph if prop_def.requires_ph else 'N/A'}")

            # Calculate property value
            if prop_def.requires_ph:
                self.logger.debug(f"Property '{prop_name}' requires pH. Using pH={self.config.ph} for calculation.")
                properties[prop_name] = prop_def.function(peptide, pH=self.config.ph)
            else:
                properties[prop_name] = prop_def.function(peptide)
            
            # Check limits
            config_constraint = getattr(self.config, prop_name)
            if config_constraint.min is not None and properties[prop_name] < config_constraint.min:
                in_limits = False
            if config_constraint.max is not None and properties[prop_name] > config_constraint.max:
                in_limits = False
            self.logger.debug(f"Calculated property '{prop_name}': {properties[prop_name]}. Limits: min={config_constraint.min}, max={config_constraint.max}. In limits: {in_limits}")

            # Calculate distance from target
            if config_constraint.target is not None:
                distance_from_target[prop_name] = abs(properties[prop_name] - config_constraint.target)
                self.logger.debug(f"Distance from target for property '{prop_name}': {distance_from_target[prop_name]} (target: {config_constraint.target})")
            else:
                self.logger.warning(f"No target defined for property '{prop_name}' in ChemistConfig. Distance from target will not be calculated for this property.")
        return properties, distance_from_target, in_limits

    def _score_property(self, peptide_value: float, config_constraint: RangeTarget) -> float:
        """
        Score a single property based on distance from target with exponential decay.
        Returns a score between 0 and 1, where:
        - Score is 0 if the value is outside acceptable limits
        - Score is based on distance decay from target if within limits
        - Decay is normalized using the acceptable range around the target
        """
        # Check if within limits

        if config_constraint.min is not None and peptide_value < config_constraint.min:
            self.logger.debug(f"Peptide value {peptide_value} is below minimum {config_constraint.min if config_constraint.min is not None else 'N/A'}. Score: 0.0")
            return 0.0
        if config_constraint.max is not None and peptide_value > config_constraint.max:
            self.logger.debug(f"Peptide value {peptide_value} is above maximum {config_constraint.max if config_constraint.max is not None else 'N/A'}. Score: 0.0")
            return 0.0
        
        # If no target defined, cannot score
        if config_constraint.target is None:
            self.logger.debug(f"No target defined for property. Peptide value: {peptide_value}. Returning default score of 1.0 since it's within limits.")
            return 1.0
        
        # Calculate distance from target
        distance = abs(peptide_value - config_constraint.target)
        
        # Calculate max acceptable distance (range from target to furthest limit)
        max_distance = 0.0
        if config_constraint.min is not None:
            max_distance = max(max_distance, abs(config_constraint.target - config_constraint.min))
        if config_constraint.max is not None:
            max_distance = max(max_distance, abs(config_constraint.target - config_constraint.max))
        
        # If at target, return perfect score
        if distance == 0:
            self.logger.debug(f"Peptide value {peptide_value} is exactly at target {config_constraint.target}. Score: 1.0")
            return 1.0
        
        # If max_distance is 0 (target equals both min and max), return 1
        if max_distance == 0:
            self.logger.warning(f"Max distance is 0 for property with target {config_constraint.target}. Returning score of 1.0 since peptide value {peptide_value} is within limits.")
            return 1.0
        
        # Normalize distance to [0, 1] range and apply exponential decay
        normalized_distance = distance / max_distance
        score = float(exp(-normalized_distance))
        self.logger.debug(f"Scoring property with peptide value {peptide_value}, target {config_constraint.target}, distance {distance}, normalized distance {normalized_distance}, score {score}")
        return score

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
        Scores are calculated per-property based on distance from target with decay, normalized within property limits.
        Scores are comparable across different batches/queries.
        Returns a list of dictionaries, each containing property scores and an overall weighted score.
        Also includes a boolean indicating if each peptide is within all limits or not.
        """
        # Validate peptides
        for peptide in peptides:
            if not self.validate_sequence(peptide):
                self.logger.warning(f"Peptide '{peptide}' is invalid and will be skipped.")
                peptides.remove(peptide)
        if not peptides:
            return []

        analyzed_peptides = {peptide: self._analyze_peptide(peptide) for peptide in peptides}

        # List of properties valid for the config (not None)
        valid_properties = [
            prop_name
            for prop_name in PROPERTY_REGISTRY.keys()
            if getattr(self.config, prop_name) is not None
        ]

        scored_peptides = []
        for peptide in analyzed_peptides.keys():
            result = analyzed_peptides[peptide]
            properties = result["properties"]
            overall_score = 0.0
            property_scores = {}

            # Score each property independently
            for prop_name in valid_properties:
                if prop_name not in properties:
                    self.logger.warning(f"Property '{prop_name}' was expected but not calculated for peptide '{peptide}'. Skipping this property for scoring.")
                    continue
                
                peptide_value = properties[prop_name]
                config_constraint = getattr(self.config, prop_name)
                
                # Get property score using distance decay method
                prop_score = self._score_property(peptide_value, config_constraint)
                property_scores[prop_name] = prop_score
                
                # Get weight for this property
                weight = config_constraint.weight if config_constraint.weight is not None else 1.0
                
                # Accumulate weighted score
                overall_score += prop_score * weight
            
            # Normalize overall score by number of properties (to keep it in reasonable range)
            num_properties = len(property_scores) if property_scores else 1
            normalized_overall_score = overall_score / num_properties
            
            scored_peptides.append(
                {
                    "sequence": peptide,
                    "properties": result["properties"],
                    "property_scores": property_scores,
                    "score": normalized_overall_score,
                    "in_limits": result["in_limits"],
                }
            )
        self.logger.debug(f"Evaluated peptides with scores: {scored_peptides}")
        return scored_peptides
    
    def get_top_filtered_peptides(self, peptides: List[str], topK: int) -> List[str]:
        peptide_evaluated = self.evaluate_peptides(peptides)
        # Sort by: in_limits peptides first (True first), then by score (descending)
        sorted_peptides = sorted(peptide_evaluated, key=lambda x: (not x["in_limits"], -x["score"]))
        self.logger.debug(f"Peptides sorted by in_limits and score: {sorted_peptides}")
        
        # Log if not enough in-limit peptides
        in_limit_count = sum(1 for p in sorted_peptides if p["in_limits"])
        if in_limit_count < topK:
            self.logger.info(f"Only {in_limit_count} peptides are within limits, returning {topK - in_limit_count} out of limits ranked by score.")
        self.logger.debug(f"Top {topK} peptides: {sorted_peptides[:topK]}")
        return sorted_peptides[:topK]





