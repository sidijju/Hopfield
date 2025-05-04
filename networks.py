import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 50257

class RecurrentBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        output, _ = self.rnn(x)        # [batch_size, seq_len, hidden_size]
        logits = self.out(output)  # [batch_size, seq_len, vocab_size]
        pooled_output = output[:, -1, :]  # [batch_size, hidden_size]
        return logits, pooled_output
    
class LinearAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # did some research and the performer paper recommends this? 
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # Compute context
        KV = torch.einsum("bnd,bne->bde", K, V)  # [batch, dim, dim]
        Z = torch.einsum("bnd,bd->bn", Q, K.sum(dim=1) + 1e-6)  # Normalization
        context = torch.einsum("bnd,bde->bne", Q, KV) / Z.unsqueeze(-1)

        return self.out(context)
    
class LinearAttentionBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)  # GPT-2 vocab size
        self.layers = nn.ModuleList([LinearAttentionLayer(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)

        token_embeddings = x
        logits = self.lm_head(token_embeddings)
        pooled_output = x[:, -1]  # Use last token for pooled_output
        # pooled_output = x.mean(dim=1) # Mean pooling

        return logits, pooled_output

class HopfieldAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Hopfield-style associative update
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attention_weights = F.softmax(self.beta * attention_scores, dim=-1)
        context = torch.bmm(attention_weights, V)

        return self.out_proj(context)

class HopfieldAttentionBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.layers = nn.ModuleList([HopfieldAttentionLayer(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)

        token_embeddings = x
        logits = self.lm_head(token_embeddings)
        pooled_output = x.mean(dim=1) # Mean pooling

        return logits, pooled_output

class NextTokenPredictor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids):
        logits, _ = self.backbone(input_ids)
        return logits
    
class SequenceClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.hidden_dim, num_classes)

    def forward(self, input_ids):
        _, pooled_output = self.backbone(input_ids)
        logits = self.classifier(pooled_output)
        return logits


