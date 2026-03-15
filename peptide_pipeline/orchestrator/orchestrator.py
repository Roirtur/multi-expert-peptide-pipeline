from typing import List, Dict, Any, Optional
from logger import get_logger

from peptide_pipeline.generator.base import BaseGenerator
from peptide_pipeline.chemist.base import BaseChemist
from peptide_pipeline.biologist.base import BaseBiologist
from peptide_pipeline.orchestrator.base import BaseOrchestrator

class Orchestrator(BaseOrchestrator):
    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        super().__init__(generator, chemist, biologist)
        self.logger = get_logger("Orchestrator")

    def run(self, 
            nb_iterations: int, 
            nb_peptides: int, 
            top_k: int, 
            exploration_rate: float = 0.1,
            initial_peptide: Optional[str] = None) -> List[Dict[str, Any]]:
        
        results: List[Dict[str, Any]] = []

        if initial_peptide:
            self.logger.info(f"Pipeline Start: Seeding with '{initial_peptide}'")
            # Premier round : on demande des variantes de ce peptide
            feedback = {"count": nb_peptides, "exploration_rate": exploration_rate}
            candidates = self.generator.modify_peptides([initial_peptide], feedback)
        else:
            self.logger.info("Pipeline Start: Random generation (No seed provided)")
            candidates = self.generator.generate_peptides(count=nb_peptides)
        
        self.logger.info(f"Running for {nb_iterations} iterations")

        for i in range(nb_iterations):
            #chem agent
            validity_mask = self.chemist.check_validity(candidates)
            valid_candidates = [c for c, v in zip(candidates, validity_mask) if v]

            if not valid_candidates:
                self.logger.warning(f"Iteration {i+1}: No valid candidates found. Restarting random generation.")
                candidates = self.generator.generate_peptides(count=nb_peptides)
                continue

            valid_props = self.chemist.calculate_properties(valid_candidates)

            #bio agent
            scores = self.biologist.score_peptides(valid_candidates)
            iteration_results = []
            for peptide, score, props in zip(valid_candidates, scores, valid_props):
                iteration_results.append({"peptide": peptide, "score": score, "properties": props})

            iteration_results.sort(key=lambda x: x["score"], reverse=True)
            
            #top-K
            top_k_pep = iteration_results[:top_k]
            
            results = top_k_pep
            best_score = results[0]['score'] if results else 0.0
            self.logger.info(f"Iteration {i+1}: Best Score = {best_score:.4f} (Valid: {len(valid_candidates)}/{len(candidates)})")

            if i < nb_iterations - 1:
                pep_seqs = [s["peptide"] for s in top_k_pep]
                
                feedback = {
                    "count": nb_peptides,
                    "exploration_rate": exploration_rate,
                    "parents_scores": [s["score"] for s in top_k_pep]
                }
                
                candidates = self.generator.modify_peptides(pep_seqs, feedback)

        return results




