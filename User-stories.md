## Researcher

- As a Researcher, I want the system to output a final "Top-K" list of the best peptides so that I have a highly curated set of candidates ready for analysis.

- As a Researcher, I want the system to automatically calculate and report metrics like validity percentage, average score, and peptide diversity so that I can objectively evaluate the quality of the generated batches.

- As a Researcher, I want to compare the multi-agent pipeline's results against a non-agentic baseline (like random generation or a simple heuristic) so that I can measure and prove the actual gain of using an iterative, multi-agent approach.

- As a Researcher, I want to access detailed logs of the pipeline's iterations so that I can trace the trajectory of improvement and understand how the agents arrived at the final solutions.

## Orchestrator

- As the Orchestrator, I want to route the Biologist agent's critiques back to the Designer agent so that the system can iterate and improve the peptides over several rounds.

- As the Orchestrator, I want to apply specific stopping criteria so that the iteration loop finishes cleanly without running infinitely once a satisfactory result (or maximum round limit) is reached.

## Chemistry Agent

- As the Chemistry Agent, I want to filter the proposed peptides against chemical constraints like length, net charge, and forbidden motifs so that only chemically valid and feasible proxy molecules proceed to biological scoring.

## Biology agent

- As the Biology Agent, I want to evaluate valid peptides using a quantitative score based on expected activity so that the candidates can be ranked objectively and better new peptide can be generated.

## Generator

- As a Generator, I want to receive specific target characteristics or a peptide as input so that I can generate an initial batch of X similar variants so that the pipeline can iteratively improve the candidates.

- As a Generator, I want to apply varied prompting, templates, and mutation operators to my input so that I produce a batch that balances accurate refinement (convergence) with necessary structural exploration.