import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from baseline.data_handler import AA_TO_IDX


class Encoder(nn.Module):
    """Encodes peptide tokens + condition vector into latent distribution params."""

    def __init__(self, vocab_size, embedding_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + condition_dim, latent_dim)

    def forward(self, x, c):
        """Return (mu, logvar) for q(z|x,c)."""
        embedded = self.embedding(x)

        _, hidden = self.gru(embedded)

        hidden = hidden.squeeze(0)

        conditioned = torch.cat((hidden, c), dim=1)
        mu = self.fc_mu(conditioned)
        logvar = self.fc_logvar(conditioned)

        return mu, logvar

class Decoder(nn.Module):
    """Autoregressive GRU decoder conditioned on latent vector and properties."""

    def __init__(self, vocab_size, condition_dim, hidden_dim, latent_dim, embedding_dim, max_seq_len, sos_idx=1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.sos_idx = sos_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc_hidden = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim + latent_dim + condition_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, z, c):
        """Build decoder context and initial GRU hidden state."""
        context = torch.cat((z, c), dim=1)
        hidden = self.fc_hidden(context).unsqueeze(0)
        return context, hidden

    def step(self, prev_tokens, context, hidden):
        """Run one autoregressive decoding step (used during generation)."""
        token_emb = self.token_embedding(prev_tokens)
        step_input = torch.cat((token_emb, context), dim=1).unsqueeze(1)
        gru_out, hidden = self.gru(step_input, hidden)
        logits = self.fc_out(gru_out.squeeze(1))
        return logits, hidden

    def forward(self, z, c, input_seq=None):
        """
        Decode logits. Uses fast vectorized operations during training 
        and token-by-token generation during raw inference.
        """
        context, hidden = self.init_state(z, c)
        batch_size = z.size(0)

        if input_seq is not None:
            seq_len = input_seq.size(1)
            
            token_embs = self.token_embedding(input_seq)
            
            context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
            
            gru_input = torch.cat((token_embs, context_expanded), dim=-1)
            gru_out, hidden = self.gru(gru_input, hidden)
            
            logits = self.fc_out(gru_out)
            return logits

        outputs = []
        prev_tokens = torch.full(
            (batch_size,),
            self.sos_idx,
            dtype=torch.long,
            device=z.device,
        )

        for _ in range(self.max_seq_len):
            logits, hidden = self.step(prev_tokens, context, hidden)
            outputs.append(logits.unsqueeze(1))
            prev_tokens = torch.argmax(logits, dim=1)

        return torch.cat(outputs, dim=1)

class PeptideCVAE(nn.Module):
    """Conditional VAE for peptide sequence generation."""

    def __init__(self, vocab_size, condition_dim, max_seq_len=14, embedding_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.sos_idx = AA_TO_IDX["<SOS>"]
        self.eos_idx = AA_TO_IDX["<EOS>"]
        self.pad_idx = AA_TO_IDX["<PAD>"]

        self.encoder = Encoder(vocab_size, embedding_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(
            vocab_size,
            condition_dim,
            hidden_dim,
            latent_dim,
            embedding_dim,
            max_seq_len,
            sos_idx=self.sos_idx,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        """Sample z via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)

        decoder_input = x[:, :-1]
        recon_x = self.decoder(z, c, input_seq=decoder_input)

        return recon_x, mu, logvar

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        """Keep only top-k logits per row; mask the rest with -inf."""
        if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
            return logits

        top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, top_idx, top_vals)
        return filtered

    def generate(self, c, num_samples=1, temperature=1.0, top_k=None):
        """
        Generate peptide token sequences conditioned on a property vector.

        c: Tensor of shape [1, condition_dim] (already scaled).
        """
        self.eval()

        with torch.no_grad():
            device = c.device
            c = c.repeat(num_samples, 1)

            z = torch.randn(num_samples, self.latent_dim).to(device)

            context, hidden = self.decoder.init_state(z, c)
            prev_tokens = torch.full(
                (num_samples,),
                self.sos_idx,
                dtype=torch.long,
                device=device,
            )

            generated_tokens = []
            steps = self.max_seq_len - 1
            temp = max(float(temperature), 1e-5)

            for _ in range(steps):
                logits, hidden = self.decoder.step(prev_tokens, context, hidden)
                logits = logits / temp

                logits = self._apply_top_k(logits, top_k)

                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                generated_tokens.append(next_tokens.unsqueeze(1))
                prev_tokens = next_tokens

            return torch.cat(generated_tokens, dim=1)

def cvae_loss_function(
    recon_x: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total CVAE loss
    """
    batch_size = recon_x.size(0)
    
    logits_for_ce = recon_x.transpose(1, 2)

    recon_loss = F.cross_entropy(
        logits_for_ce, 
        targets, 
        ignore_index=AA_TO_IDX["<PAD>"], 
        reduction='sum'
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = (recon_loss + beta * kl_loss) / batch_size
    
    return total_loss, recon_loss / batch_size, kl_loss / batch_size