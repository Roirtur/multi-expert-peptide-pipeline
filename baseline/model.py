import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        # 1. Embed the integer sequences into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Process the sequence with a GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # 3. Linear layers to map the concatenated (hidden_state + conditions) to latent space
        self.fc_mu = nn.Linear(hidden_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + condition_dim, latent_dim)

    def forward(self, x, c):
        """
        x: [batch_size, seq_len] - The input peptide sequence (encoded as integers)
        c: [batch_size, condition_dim] - The scaled chemical properties
        """
        # Get embeddings: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # Pass through GRU. We only care about the final hidden state to summarize the sequence.
        _, hidden = self.gru(embedded)
        
        # hidden shape is [1, batch_size, hidden_dim], squeeze it to [batch_size, hidden_dim]
        hidden = hidden.squeeze(0)
        
        # Condition the latent space by concatenating the hidden state with the properties
        # shape: [batch_size, hidden_dim + condition_dim]
        concat = torch.cat((hidden, c), dim=1)
        
        # Output the mean and log variance for the latent distribution
        mu = self.fc_mu(concat)
        logvar = self.fc_logvar(concat)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, vocab_size, condition_dim, hidden_dim, latent_dim, embedding_dim, max_seq_len, sos_idx=1):
        super(Decoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.sos_idx = sos_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 1. Map the concatenated (z + conditions) back to a hidden state for the GRU
        self.fc_hidden = nn.Linear(latent_dim + condition_dim, hidden_dim)
        
        # 2. GRU to generate sequence tokens.
        # We feed token embeddings plus conditioning context at each step.
        self.gru = nn.GRU(embedding_dim + latent_dim + condition_dim, hidden_dim, batch_first=True)
        
        # 3. Final layer to map GRU output to vocabulary probabilities
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, z, c):
        """Build a fixed conditioning context and initial hidden state."""
        context = torch.cat((z, c), dim=1)
        hidden = self.fc_hidden(context).unsqueeze(0)
        return context, hidden

    def step(self, prev_tokens, context, hidden):
        """Run one autoregressive decoding step."""
        token_emb = self.token_embedding(prev_tokens)
        step_input = torch.cat((token_emb, context), dim=1).unsqueeze(1)
        gru_out, hidden = self.gru(step_input, hidden)
        logits = self.fc_out(gru_out.squeeze(1))
        return logits, hidden

    def forward(self, z, c, input_seq=None):
        """
        z: [batch_size, latent_dim] - The sampled latent vector
        c: [batch_size, condition_dim] - The scaled chemical properties
        input_seq: [batch_size, seq_len] - Optional teacher-forced decoder input tokens
        """
        context, hidden = self.init_state(z, c)
        batch_size = z.size(0)
        steps = input_seq.size(1) if input_seq is not None else self.max_seq_len

        outputs = []
        prev_tokens = torch.full(
            (batch_size,),
            self.sos_idx,
            dtype=torch.long,
            device=z.device,
        )

        for t in range(steps):
            if input_seq is not None:
                prev_tokens = input_seq[:, t]
            logits, hidden = self.step(prev_tokens, context, hidden)
            outputs.append(logits.unsqueeze(1))
            if input_seq is None:
                prev_tokens = torch.argmax(logits, dim=1)

        return torch.cat(outputs, dim=1)

class PeptideCVAE(nn.Module):
    def __init__(self, vocab_size, condition_dim, max_seq_len=14, embedding_dim=64, hidden_dim=128, latent_dim=32):
        """
        Args:
            vocab_size (int): Size of the vocabulary (amino acids + special tokens).
            condition_dim (int): Number of conditioning variables (e.g., 8).
            max_seq_len (int): Max peptide length + 2 (for <SOS> and <EOS>). If dataset max_len is 12, this is 14.
            embedding_dim (int): Size of amino acid embeddings.
            hidden_dim (int): Hidden size for GRU layers.
            latent_dim (int): Size of the bottleneck latent space (z).
        """
        super(PeptideCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.sos_idx = 1
        self.eos_idx = 2
        self.pad_idx = 0
        
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

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick to sample z from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # 1. Encode into latent space distributions
        mu, logvar = self.encoder(x, c)
        
        # 2. Sample from the distribution
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode back into a sequence using teacher forcing
        decoder_input = x[:, :-1]
        recon_x = self.decoder(z, c, input_seq=decoder_input)
        
        return recon_x, mu, logvar
        
    def generate(self, c, num_samples=1, temperature=1.0, top_k=None):
        """
        Utility method to generate novel peptides purely from chemical conditions.
        c: Tensor of scaled properties [1, condition_dim]
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            device = c.device
            # Ensure condition vector matches the number of requested samples
            c = c.repeat(num_samples, 1)
            
            # Sample pure noise from a standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)

            context, hidden = self.decoder.init_state(z, c)
            prev_tokens = torch.full(
                (num_samples,),
                self.sos_idx,
                dtype=torch.long,
                device=device,
            )

            generated = []
            steps = self.max_seq_len - 1
            temp = max(float(temperature), 1e-5)

            for _ in range(steps):
                logits, hidden = self.decoder.step(prev_tokens, context, hidden)
                logits = logits / temp

                if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                    top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
                    filtered = torch.full_like(logits, float('-inf'))
                    filtered.scatter_(1, top_idx, top_vals)
                    logits = filtered

                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                generated.append(next_tokens.unsqueeze(1))
                prev_tokens = next_tokens

            predicted_sequences = torch.cat(generated, dim=1)
            return predicted_sequences

def cvae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the loss: Reconstruction Loss (CrossEntropy) + beta * KL Divergence
    
    recon_x: [batch_size, target_seq_len, vocab_size] (Logits from Decoder)
    x: [batch_size, target_seq_len] (Target integers)
    """
    # CrossEntropyLoss expects logits of shape [batch_size, vocab_size, max_seq_len]
    recon_x = recon_x.transpose(1, 2)
    
    # Reconstruction loss (categorical cross-entropy)
    # Ignore the padding token when calculating loss (assuming <PAD> is index 0)
    recon_loss = F.cross_entropy(recon_x, x, ignore_index=0, reduction='sum')
    
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Combine with the beta weight (useful for beta-VAE/KL annealing)
    total_loss = recon_loss + beta * kl_loss
    
    # Return all losses for monitoring
    return total_loss, recon_loss, kl_loss