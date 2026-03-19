import random
from typing import List, Dict, Any, Optional

from peptide_pipeline.generator.base import BaseGenerator
from peptide_pipeline.chemist.base import BaseChemist
from peptide_pipeline.biologist.base import BaseBiologist
from peptide_pipeline.orchestrator.base import BaseOrchestrator

class Orchestrator(BaseOrchestrator):
    def __init__(self, generator: BaseGenerator, chemist: BaseChemist, biologist: BaseBiologist):
        super().__init__(generator, chemist, biologist)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _infer_target_from_chemist_config(self) -> Dict[str, Any]:
        inferred: Dict[str, Any] = {}
        config = getattr(self.chemist, "config", None)
        if config is None:
            return inferred

        mapping = {
            "length": "size",
            "molecular_weight": "molecular_weight",
            "net_charge": "net_charge_pH5_5",
            "isoelectric_point": "isoelectric_point",
            "hydrophobicity": "hydrophobicity",
            "cathionicity": "cathionicity",
            "logp": "logp",
        }
        for chemist_name, generator_name in mapping.items():
            rt = getattr(config, chemist_name, None)
            target_value = getattr(rt, "target", None) if rt is not None else None
            if target_value is not None:
                inferred[generator_name] = target_value

        return inferred

    def _normalize_target_constraints(self, final_target: Dict[str, Any]) -> Dict[str, Any]:
        alias = {
            "length": "size",
            "net_charge": "net_charge_pH5_5",
        }
        constraints: Dict[str, Any] = {}
        for k, v in (final_target or {}).items():
            target_key = alias.get(k, k)
            constraints[target_key] = v
        return constraints

    def _build_parent_constraints(
        self,
        parent_entry: Dict[str, Any],
        base_constraints: Dict[str, Any],
        blend: float = 0.5,
    ) -> Dict[str, Any]:
        constraints = dict(base_constraints)
        props = parent_entry.get("properties", {}) or {}

        prop_to_constraint = {
            "length": "size",
            "molecular_weight": "molecular_weight",
            "net_charge": "net_charge_pH5_5",
            "isoelectric_point": "isoelectric_point",
            "hydrophobicity": "hydrophobicity",
            "cathionicity": "cathionicity",
            "logp": "logp",
        }

        for prop_name, constraint_key in prop_to_constraint.items():
            if prop_name not in props:
                continue
            if constraint_key not in constraints:
                continue
            base_value = self._safe_float(constraints[constraint_key])
            parent_value = self._safe_float(props[prop_name], default=base_value)
            constraints[constraint_key] = (1.0 - blend) * base_value + blend * parent_value

        return constraints

    def _chemist_relative_score(self, chemist_result: Dict[str, Any], final_target: Dict[str, Any]) -> float:
        properties = chemist_result.get("properties", {}) or {}
        target_to_prop = {
            "size": "length",
            "molecular_weight": "molecular_weight",
            "net_charge_pH5_5": "net_charge",
            "isoelectric_point": "isoelectric_point",
            "hydrophobicity": "hydrophobicity",
            "cathionicity": "cathionicity",
            "logp": "logp",
            "length": "length",
            "net_charge": "net_charge",
        }

        scores: List[float] = []
        for target_key, target_value in (final_target or {}).items():
            prop_name = target_to_prop.get(target_key)
            if prop_name is None or prop_name not in properties:
                continue

            tv = self._safe_float(target_value)
            pv = self._safe_float(properties[prop_name])
            denom = max(abs(tv), 1.0)
            rel_error = abs(pv - tv) / denom
            scores.append(1.0 / (1.0 + rel_error))

        if not scores:
            return self._safe_float(chemist_result.get("score", 0.0))

        return sum(scores) / len(scores)

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
        if final_target is None:
            final_target = self._infer_target_from_chemist_config()
            if final_target:
                self.logger.warning("No final_target provided. Using targets inferred from chemist config.")
            else:
                self.logger.warning("No final_target provided and no targets inferred from chemist config. Generation will use default generator constraints.")

        base_constraints = self._normalize_target_constraints(final_target)

        self.logger.info(
            f"Pipeline start: {nb_iterations} iterations, {nb_peptides} peptides/iteration, "
            f"exploration_rate={exploration_rate:.2f}, top_k={top_k}"
        )

        global_best_by_sequence: Dict[str, Dict[str, Any]] = {}
        global_pool: List[Dict[str, Any]] = []

        for i in range(nb_iterations):
            iteration_idx = i + 1
            exploration_hit = i > 0 and random.random() < exploration_rate

            if i == 0:
                if initial_peptide:
                    self.logger.info(
                        f"Iteration {iteration_idx}: initial_peptide is provided ('{initial_peptide}') but generation is driven by final_target constraints only."
                    )
                candidates = self.generator.generate_peptides(count=nb_peptides, constraints=base_constraints)
                parent_mode = "target_only"
            else:
                if not global_pool:
                    self.logger.warning(
                        f"Iteration {iteration_idx}: Global ranking is empty, fallback to final_target-only generation."
                    )
                    candidates = self.generator.generate_peptides(count=nb_peptides, constraints=base_constraints)
                    parent_mode = "fallback_target_only"
                else:
                    effective_parent_count = max(1, min(random_parent_count, len(global_pool), nb_peptides))
                    if exploration_hit:
                        selected_parents = random.sample(global_pool, k=effective_parent_count)
                        parent_mode = "exploration_random"
                    else:
                        selected_parents = global_pool[:effective_parent_count]
                        parent_mode = "exploitation_top"

                    base_batch = nb_peptides // effective_parent_count
                    remainder = nb_peptides % effective_parent_count
                    per_parent_counts = [base_batch + (1 if idx < remainder else 0) for idx in range(effective_parent_count)]

                    candidates = []
                    for parent, local_count in zip(selected_parents, per_parent_counts):
                        if local_count <= 0:
                            continue
                        blended_constraints = self._build_parent_constraints(
                            parent_entry=parent,
                            base_constraints=base_constraints,
                            blend=0.5,
                        )
                        generated = self.generator.generate_peptides(count=local_count, constraints=blended_constraints)
                        candidates.extend(generated)

            generated_count = len(candidates)
            if generated_count == 0:
                self.logger.warning(f"Iteration {iteration_idx}: Generator returned no candidates.")
                continue

            chemist_results = self.chemist.evaluate_peptides(candidates)
            if not chemist_results:
                self.logger.warning(f"Iteration {iteration_idx}: Chemist returned no evaluable candidates.")
                continue

            in_limit_candidates = [c for c in chemist_results if bool(c.get("in_limits", False))]
            selected_candidates = in_limit_candidates if in_limit_candidates else chemist_results

            if not in_limit_candidates:
                self.logger.warning(
                    f"Iteration {iteration_idx}: No in-limit candidates. Using out-of-limit set for biological scoring."
                )

            valid_candidates = [c.get("sequence", "") for c in selected_candidates if c.get("sequence")]
            selected_candidates = [c for c in selected_candidates if c.get("sequence")]

            if not valid_candidates:
                self.logger.warning(f"Iteration {iteration_idx}: No valid sequences after chemist filtering.")
                continue

            bio_context = None
            if "target_sequence" in final_target:
                bio_context = final_target["target_sequence"]
            elif "reference_peptide" in final_target:
                bio_context = final_target["reference_peptide"]

            if bio_context is not None:
                bio_scores = self.biologist.predict_activity(valid_candidates, context=bio_context)
            else:
                bio_scores = self.biologist.score_peptides(valid_candidates)

            iteration_results: List[Dict[str, Any]] = []
            for candidate, bio_score in zip(selected_candidates, bio_scores):
                chem_score = self._chemist_relative_score(candidate, final_target)
                bio_score_float = self._safe_float(bio_score, default=0.0)
                combined_score = (chem_score + bio_score_float) / 2.0

                row = {
                    "peptide": candidate["sequence"],
                    "score": combined_score,
                    "combined_score": combined_score,
                    "chemist_score": chem_score,
                    "biologist_score": bio_score_float,
                    "in_limits": bool(candidate.get("in_limits", False)),
                    "properties": candidate.get("properties", {}),
                    "iteration": iteration_idx,
                }
                iteration_results.append(row)

                existing = global_best_by_sequence.get(row["peptide"])
                if existing is None or row["combined_score"] > existing["combined_score"]:
                    global_best_by_sequence[row["peptide"]] = row

            global_pool = sorted(
                global_best_by_sequence.values(),
                key=lambda x: (x["in_limits"], x["combined_score"], x["biologist_score"], x["chemist_score"]),
                reverse=True,
            )

            iteration_results.sort(
                key=lambda x: (x["in_limits"], x["combined_score"], x["biologist_score"], x["chemist_score"]),
                reverse=True,
            )

            best_combined = iteration_results[0]["combined_score"] if iteration_results else 0.0

            in_limits_count = len(in_limit_candidates)
            off_limits_count = len(chemist_results) - in_limits_count

            self.logger.info(
                f"Iteration {iteration_idx}: mode={parent_mode}, generated={generated_count}, "
                f"in_limits={in_limits_count}, off_limits={off_limits_count}, "
                f"best_combined={best_combined:.4f}, global_unique={len(global_pool)}"
            )

        final_ranking = sorted(
            global_best_by_sequence.values(),
            key=lambda x: (x["in_limits"], x["combined_score"], x["biologist_score"], x["chemist_score"]),
            reverse=True,
        )

        final_top_k = final_ranking[:top_k]

        self.logger.info(
            f"Pipeline finished: global_unique={len(global_best_by_sequence)}, "
            f"returned_top_k={len(final_top_k)}"
        )

        return final_top_k




