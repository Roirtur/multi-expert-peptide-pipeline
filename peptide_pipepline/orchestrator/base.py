from typing import List, Dict, Any

from peptide_pipepline.generator.base import BaseGenerator
from peptide_pipepline.chemist.base import BaseChemist
from peptide_pipepline.biologist.base import BaseBiologist

class BaseOrchestrator:
    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        self.generator = generator
        self.chemist = chemist
        self.biologist = biologist

    def run(self, iterations=10, population_size=100, top_k=10):
        # 1. Generate initial population with Generator Agent
        population = self.generator.generate_peptides(count=population_size)

        for i in range(iterations):
            # 2. Filter valid peptides with Chemist Agent
            validity_mask = self.chemist.check_validity(population)
            valid_population = [
                pop for pop, valid in zip(population, validity_mask) 
                if valid
            ]

            if not valid_population:
                break

            # 3. Score candidates with Biologist Agent (Batch)
            # Assuming predict_activity returns a list of scores
            scores = self.biologist.predict_activity(valid_population, "target_description")
            
            scored_population = []
            for peptide, score in zip(valid_population, scores):
                scored_population.append({"sequence": peptide, "score": score})

            # 4. Keep Top-K best candidates
            scored_population.sort(key=lambda x: x["score"], reverse=True)
            best_candidates = scored_population[:top_k]

            # 5. Generate new candidates based on best ones
            best_sequences = [c["sequence"] for c in best_candidates]
            new_candidates = self.generator.modify_peptides(best_sequences, feedback={"count": population_size - top_k})
            
            # Update population (Top-K + New)
            population = best_sequences + new_candidates

        return best_candidates

