from typing import List, Dict, Any, Optional

from peptide_pipeline.generator.base import BaseGenerator
from peptide_pipeline.chemist.base import BaseChemist
from peptide_pipeline.biologist.base import BaseBiologist
from peptide_pipeline.orchestrator.base import BaseOrchestrator

class Orchestrator(BaseOrchestrator):
    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        super().__init__(generator, chemist, biologist)

    def run(self, 
            nb_iterations: int, 
            nb_peptides: int, 
            top_k: int, 
            exploration_rate: float = 0.1,
            initial_peptide: Optional[str] = None) -> List[Dict[str, Any]]:
        
        """
        Main Orchestrator Agent Loop, responsible for maintaining the whole pipeline iteration process
        """
        
        results: List[Dict[str, Any]] = []

        if initial_peptide:
            self.logger.info(f"Pipeline Start: Seeding with '{initial_peptide}'")
            feedback = {"count": nb_peptides, "exploration_rate": exploration_rate}
            candidates = self.generator.modify_peptides([initial_peptide], feedback)
        else:
            self.logger.info("Pipeline Start: Random generation (No seed provided)")
            candidates = self.generator.generate_peptides(count=nb_peptides)
        
        self.logger.info(f"Running for {nb_iterations} iterations")

        for i in range(nb_iterations):

            #get peptides properties and filter the invalid ones at the same time
            chemist_results = self.chemist.evaluate_peptides(candidates)
            if not chemist_results:
                self.logger.warning(f"Iteration {i+1}: No chemically evaluable candidates found. Restarting random generation.")
                candidates = self.generator.generate_peptides(count=nb_peptides)
                continue

            in_limit_candidates = [c for c in chemist_results if c.get("in_limits") is True]
            selected_candidates = in_limit_candidates if in_limit_candidates else chemist_results                

            if not in_limit_candidates:
                self.logger.warning(
                    f"Iteration {i+1}: No candidates within chemistry limits. Falling back to out-of-limit candidates ranked by bio score."
                )

            #prepare data for bio agent
            valid_candidates = [c["sequence"] for c in selected_candidates]
            valid_props = [c.get("properties", {}) for c in selected_candidates]
            chem_scores = [float(c.get("score", 0.0)) for c in selected_candidates]
            chem_in_limits = [bool(c.get("in_limits", False)) for c in selected_candidates]

            # Bio agent
            scores = self.biologist.score_peptides(valid_candidates)
            iteration_results = []
            for peptide, score, props, chem_score, in_limits in zip(
                valid_candidates,
                scores,
                valid_props,
                chem_scores,
                chem_in_limits,
            ):
                iteration_results.append(
                    {
                        "peptide": peptide,
                        "score": score,
                        "chemist_score": chem_score,
                        "in_limits": in_limits,
                        "properties": props,
                    }
                )

            iteration_results.sort(
                key=lambda x: (x["in_limits"], x["score"], x["chemist_score"]),
                reverse=True,
            )
            
            top_k_pep = iteration_results[:top_k]
            
            results = top_k_pep
            best_score = results[0]['score'] if results else 0.0
            in_limit_count = sum(1 for r in iteration_results if r["in_limits"])
            self.logger.info(
                f"Iteration {i+1}: Best Bio Score = {best_score:.4f} "
                f"(Selected: {len(valid_candidates)}/{len(candidates)}, In-limits: {in_limit_count})"
            )

            if i < nb_iterations - 1:
                pep_seqs = [s["peptide"] for s in top_k_pep]
                
                feedback = {
                    "count": nb_peptides,
                    "exploration_rate": exploration_rate,
                    "parents_scores": [s["score"] for s in top_k_pep]
                }
                
                candidates = self.generator.modify_peptides(pep_seqs, feedback)

        return results




