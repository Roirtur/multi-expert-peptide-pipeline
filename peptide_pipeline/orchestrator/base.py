import logging
from typing import List, Dict, Any, Optional

from peptide_pipeline.generator.base import BaseGenerator
from peptide_pipeline.chemist.base import BaseChemist
from peptide_pipeline.biologist.base import BaseBiologist


class BaseOrchestrator:
    logger = logging.getLogger("peptide_pipeline.orchestrator")

    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        self.generator = generator
        self.chemist = chemist
        self.biologist = biologist

    def run(
        self,
        nb_iterations: int,
        nb_peptides: int,
        top_k: int,
        exploration_rate: float = 0.1,
        initial_peptide: Optional[str] = None,
        final_target: Optional[Dict[str, Any]] = None,
        random_parent_count: int = 4,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement run with orchestrator loop logic.")

    def get_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement get_metrics.")

