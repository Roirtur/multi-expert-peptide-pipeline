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
    def __init__(self, vocab_size, condition_dim, hidden_dim, latent_dim, max_seq_len):
        super(Decoder, self).__init__()
        self.max_seq_len = max_seq_len
        
        # 1. Map the concatenated (z + conditions) back to a hidden state for the GRU
        self.fc_hidden = nn.Linear(latent_dim + condition_dim, hidden_dim)
        
        # 2. GRU to generate the sequence. 
        # We'll feed the (z + conditions) vector at every time step to keep it strongly conditioned.
        self.gru = nn.GRU(latent_dim + condition_dim, hidden_dim, batch_first=True)
        
        # 3. Final layer to map GRU output to vocabulary probabilities
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, c):
        """
        z: [batch_size, latent_dim] - The sampled latent vector
        c: [batch_size, condition_dim] - The scaled chemical properties
        """
        # Concatenate latent vector with conditions: [batch_size, latent_dim + condition_dim]
        concat = torch.cat((z, c), dim=1)
        
        # Initialize the GRU hidden state: [1, batch_size, hidden_dim]
        hidden = self.fc_hidden(concat).unsqueeze(0)
        
        # Create a sequence of inputs for the GRU by repeating the concatenated vector
        # shape: [batch_size, max_seq_len, latent_dim + condition_dim]
        repeated_input = concat.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        
        # Pass through GRU
        gru_out, _ = self.gru(repeated_input, hidden)
        
        # Get vocabulary logits for each step: [batch_size, max_seq_len, vocab_size]
        out = self.fc_out(gru_out)
        
        return out

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
        
        self.encoder = Encoder(vocab_size, embedding_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, condition_dim, hidden_dim, latent_dim, max_seq_len)

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
        
        # 3. Decode back into a sequence
        recon_x = self.decoder(z, c)
        
        return recon_x, mu, logvar
        
    def generate(self, c, num_samples=1):
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
            
            # Decode to get logits
            logits = self.decoder(z, c)
            
            # Convert logits to probabilities and pick the highest probability token
            probs = F.softmax(logits, dim=-1)
            predicted_sequences = torch.argmax(probs, dim=-1) # [num_samples, max_seq_len]
            
            return predicted_sequences

def cvae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the loss: Reconstruction Loss (CrossEntropy) + beta * KL Divergence
    
    recon_x: [batch_size, max_seq_len, vocab_size] (Logits from Decoder)
    x: [batch_size, max_seq_len] (Target integers)
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