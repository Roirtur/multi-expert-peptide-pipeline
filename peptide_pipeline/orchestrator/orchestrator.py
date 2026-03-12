from typing import List, Dict, Any

from peptide_pipeline.generator.base import BaseGenerator
from peptide_pipeline.chemist.base import BaseChemist
from peptide_pipeline.biologist.base import BaseBiologist
from peptide_pipeline.orchestrator import BaseOrchestrator

class Orchestrator(BaseOrchestrator):
    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        self.generator = generator
        self.chemist = chemist
        self.biologist = biologist

    def run(self, nb_iterations: int, nb_peptides: int, top_k: int, exploration_rate: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        #TODO : exploration rate and actualise candidates

        initial_candidates = self.generator.generate_peptides(count=nb_peptides)

        for _ in range(nb_iterations):
            #filter + get props with chemist agent
            valid_candidates = self.chemist.check_validity(initial_candidates)
            if not valid_candidates:
                break

            valid_candidates_props = self.chemist.calculate_properties(valid_candidates)

            #get scores with bioagent
            scores = self.biologist.score_peptides(valid_candidates)

            #compute results
            results = []
            for peptide, score, props in zip(valid_candidates, scores, valid_candidates_props):
                results.append({"peptide": peptide, "score": score, "properties": props})

            #top-k
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]


        return results




