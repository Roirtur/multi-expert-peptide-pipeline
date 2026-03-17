# VAE Generator

This document explains the implementation and usage of the VAE-based peptide generator defined in [`peptide_pipeline.generator.vae_generator.VAEGenerator`](vae_generator.py).

## Overview

[`peptide_pipeline.generator.vae_generator.VAEGenerator`](vae_generator.py) is a simple Variational Autoencoder used to generate fixed-length peptide sequences over the 20 standard amino acids.

It inherits from [`peptide_pipeline.generator.base.BaseGenerator`](base.py) and implements the required methods:

- [`peptide_pipeline.generator.vae_generator.VAEGenerator.generate_peptides`](vae_generator.py)
- [`peptide_pipeline.generator.vae_generator.VAEGenerator.modify_peptides`](vae_generator.py)
- [`peptide_pipeline.generator.vae_generator.VAEGenerator.train_model`](vae_generator.py)

A usage example exists in [peptide_pipeline/generator/vae_tests.ipynb](vae_tests.ipynb).

---

## Architecture

The model is composed of two dense networks:

- **Encoder**: maps a flattened one-hot peptide representation into latent distribution parameters
- **Decoder**: maps a sampled latent vector back to logits over amino acids at each sequence position

### Input representation

Each peptide is encoded as a flattened one-hot tensor:

- 20 dimensions per sequence position
- total input size = `sequence_length * 20`

For example:

- sequence length = `6`
- input dimension = `6 * 20 = 120`

This matches the default constructor in [`peptide_pipeline.generator.vae_generator.VAEGenerator`](vae_generator.py):

- `input_dim=120`
- `latent_dim=64`
- `hidden_dim=256`

---

## Important variables

### Model hyperparameters

#### `input_dim`

Total flattened size of the peptide input.

Formula:

$$
input\_dim = L \times 20
$$

where $L$ is peptide length.

This implies the current model only supports **fixed-length peptides**.

#### `latent_dim`

Size of the latent vector $z$ sampled by the VAE.

- Smaller values compress more aggressively
- Larger values increase capacity but may reduce regularization

#### `hidden_dim`

Width of the encoder and decoder hidden layers.

This controls model capacity and training cost.

#### `self.device`

Selected compute device:

- CUDA if available
- CPU otherwise

Initialized in [`peptide_pipeline.generator.vae_generator.VAEGenerator.__init__`](vae_generator.py).

---

## Internal methods

### [`peptide_pipeline.generator.vae_generator.VAEGenerator._reparameterize`](vae_generator.py)

Implements the VAE reparameterization trick:

$$
z = \mu + \epsilon \cdot \exp(0.5 \cdot \log \sigma^2)
$$

with $\epsilon \sim \mathcal{N}(0, I)$.

Purpose:

- allows stochastic latent sampling
- keeps the model differentiable during training

---

### [`peptide_pipeline.generator.vae_generator.VAEGenerator.forward`](vae_generator.py)

Training-time forward pass:

1. encodes input `x`
2. splits encoder output into `mu` and `log_var`
3. samples latent vector `z`
4. decodes `z` into reconstruction logits

Returns:

- reconstructed logits
- `mu`
- `log_var`

---

### [`peptide_pipeline.generator.vae_generator.VAEGenerator.generate_peptides`](vae_generator.py)

Generation-time sampling:

1. samples random latent vectors
2. decodes them into logits
3. reshapes logits into `[count, num_positions, 20]`
4. applies softmax with optional temperature
5. samples one amino acid per position
6. converts sampled indices back into peptide strings

Important notes:

- generation is **position-wise**, not autoregressive
- all generated peptides have the same length
- the `constraints` argument is currently accepted but not used

---

### [`peptide_pipeline.generator.vae_generator.VAEGenerator.modify_peptides`](vae_generator.py)

Current behavior:

- ignores the input peptides
- ignores `feedback`
- simply calls [`generate_peptides`](vae_generator.py) with the same number of outputs

This satisfies the base interface from [`peptide_pipeline.generator.base.BaseGenerator`](base.py), but it is not yet a true refinement operator.

---

### [`peptide_pipeline.generator.vae_generator.VAEGenerator.train_model`](vae_generator.py)

Training loop:

1. wraps tensor data in a `TensorDataset`
2. iterates through mini-batches
3. computes reconstruction loss with cross-entropy
4. computes KL divergence
5. combines both with KL annealing
6. clips gradients
7. updates weights with Adam
8. updates learning rate with cosine annealing

Loss used:

$$
\mathcal{L} = \mathcal{L}_{recon} + \lambda_{KL} \mathcal{L}_{KL}
$$

where:

- $\mathcal{L}_{recon}$ is cross-entropy over amino acids
- $\mathcal{L}_{KL}$ regularizes the latent space
- $\lambda_{KL}$ increases gradually during training

---

### [`peptide_pipeline.generator.vae_generator.VAEGenerator._peptides_to_one_hot`](vae_generator.py)

Utility to convert peptide strings into flattened one-hot vectors.

Expected behavior:

- each peptide position occupies a block of 20 values
- one amino acid is set to 1 in each occupied position
- output shape is `[num_peptides, input_dim]`

Important limitation:

- no explicit length validation against `input_dim // 20`
- longer peptides may overflow logical expectations
- shorter peptides are implicitly zero-padded

---

## Data expectations

This generator expects peptide data already transformed into one-hot vectors.

The notebook [peptide_pipeline/generator/vae_tests.ipynb](vae_tests.ipynb) shows the expected preprocessing flow:

- define a fixed sequence length
- generate or collect peptides of that exact length
- encode them into tensors of shape `[N, L * 20]`
- train the model with [`train_model`](vae_generator.py)

This is different from the baseline CVAE pipeline in [baseline/](../../baseline), which uses tokenized sequences and conditioning vectors instead of flattened one-hot inputs.

---

## Current design choices

### Strengths

- simple and fast to train
- easy to understand
- easy to sample from
- useful as a lightweight baseline generator

### Limitations

- fixed-length sequences only
- no conditioning on chemical or biological properties
- no use of `constraints`
- no true peptide editing in [`modify_peptides`](vae_generator.py)
- decoder does not model sequential dependencies explicitly
- no checkpoint save/load helpers
- no validation metrics beyond printed losses

---

## Recommended improvements

### 1. Use constraints during generation

The `constraints` parameter in [`generate_peptides`](vae_generator.py) is currently unused.

Possible extensions:

- enforce peptide length
- bias toward charged or hydrophobic residues
- reject invalid outputs by chemist rules
- integrate with [`peptide_pipeline.orchestrator.base.BaseOrchestrator`](../orchestrator/base.py)

---

### 2. Make `modify_peptides` actually refine peptides

Current implementation ignores inputs.

Better options:

- encode existing peptides into latent space
- perturb latent vectors slightly
- decode nearby variants
- use `feedback` to control exploration strength

This would align better with the generator user story in [User-stories.md](../../User-stories.md).

---

### 3. Support variable-length peptides

Right now, peptide length is hard-coded via `input_dim`.

Possible approaches:

- add an `<EOS>` token and switch to sequence decoding
- use an autoregressive decoder
- pad sequences and mask unused positions

The baseline sequence model in [baseline/model.py](../../baseline/model.py) already shows a token-based decoder design that could inspire this change.

---

### 5. Add model persistence helpers

Recommended additions:

- `save_model(path)`
- `load_model(path)`
- save config metadata (`input_dim`, `latent_dim`, `hidden_dim`)

This would simplify reuse outside notebooks like [peptide_pipeline/generator/vae_tests.ipynb](vae_tests.ipynb).

---

### 6. Track better training metrics

Current training prints only reconstruction and KL values.

Useful additions:

- validation split
- reconstruction accuracy per residue
- uniqueness of generated peptides
- novelty against training set
- invalid residue rate
- diversity metrics

Some novelty/diversity checks are already demonstrated in [peptide_pipeline/generator/vae_tests.ipynb](vae_tests.ipynb).

---

## Example workflow

1. Prepare fixed-length peptide strings
2. Convert them with [`_peptides_to_one_hot`](vae_generator.py) or equivalent preprocessing
3. Train with [`train_model`](vae_generator.py)
4. Sample new peptides with [`generate_peptides`](vae_generator.py)
5. Pass outputs through chemist and biologist filters in the wider pipeline
