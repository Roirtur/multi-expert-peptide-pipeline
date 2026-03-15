from typing import List, Dict, Any
from logger import get_logger

from peptide_pipeline.generator.base import BaseGenerator
from peptide_pipeline.chemist.base import BaseChemist
from peptide_pipeline.biologist.base import BaseBiologist
from peptide_pipeline.orchestrator.base import BaseOrchestrator

class Orchestrator(BaseOrchestrator):
    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        super().__init__(generator, chemist, biologist)
        self.logger = get_logger("Orchestrator")

    def run(self, nb_iterations: int, nb_peptides: int, top_k: int, exploration_rate: float = 0.0) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        #Initial population generation
        candidates = self.generator.generate_peptides(count=nb_peptides)
        
        self.logger.info(f"Pipeline Start: {nb_iterations} iterations, {nb_peptides} peptides")

        for i in range(nb_iterations):
            #Chemist agent
            validity_mask = self.chemist.check_validity(candidates)
            
            valid_candidates = [
                cand for cand, is_valid in zip(candidates, validity_mask) 
                if is_valid
            ]

            if not valid_candidates:
                self.logger.warning(f"Iteration {i+1}: No valid candidates found. Stopping.")
                break

            valid_candidates_props = self.chemist.calculate_properties(valid_candidates)

            #biologist agent
            scores = self.biologist.score_peptides(valid_candidates)

            iteration_results = []
            for peptide, score, props in zip(valid_candidates, scores, valid_candidates_props):
                iteration_results.append({"peptide": peptide, "score": score, "properties": props})

            #top-k Selection
            iteration_results.sort(key=lambda x: x["score"], reverse=True)
            
            remaining = iteration_results[:top_k]
            
            results = remaining
            best_score = results[0]['score'] if results else 0.0
            self.logger.info(f"Iteration {i+1}: Best Score = {best_score:.4f} (Valid: {len(valid_candidates)}/{len(candidates)})")

            if i < nb_iterations - 1:
                survivor_sequences = [res["peptide"] for res in remaining]
                
                nb_to_generate = nb_peptides - len(survivor_sequences)
                
                if nb_to_generate > 0:
                    feedback = {
                        "count": nb_to_generate,
                        "exploration_rate": exploration_rate, #give it to generator 
                        "best_scores": [res["score"] for res in remaining]
                    }
                    new_candidates = self.generator.modify_peptides(survivor_sequences, feedback)
                    
                    candidates = survivor_sequences + new_candidates
                else:
                    candidates = survivor_sequences

        return results




