import torch
import torch.nn as nn
import torch.nn.functional as F

class SentencePairClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None, dropout=0.2):
        super(SentencePairClassifier, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def attention(self, x1, x2):
        score = torch.matmul(x1, x2.transpose(1, 2)) 
        weight1 = F.softmax(score, dim=2)             
        weight2 = F.softmax(score.transpose(1, 2), dim=2)  

        align1 = torch.matmul(weight1, x2)
        align2 = torch.matmul(weight2, x1)
        return align1, align2

    def submul(self, x1, x2):
        return torch.cat([x1, x2, x1 - x2, x1 * x2], dim=-1)

    def apply_pooling(self, x):
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        return torch.cat([avg_pool, max_pool], dim=-1)

    def forward(self, seq1, seq2):
        embed1 = self.embedding(seq1)
        embed2 = self.embedding(seq2)

        out1, _ = self.encoder(embed1)
        out2, _ = self.encoder(embed2)

        align1, align2 = self.attention(out1, out2)

        m1 = self.submul(out1, align1)
        m2 = self.submul(out2, align2)

        v1 = self.apply_pooling(m1)
        v2 = self.apply_pooling(m2)

        final = self.fc(torch.cat([v1, v2], dim=-1))
        return final
