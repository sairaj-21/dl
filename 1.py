import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

corpus = """Natural language processing enables computers to understand and interpret human language. It involves various tasks such as text classification, sentiment analysis, machine translation, and more. Deep learning techniques have significantly improved the performance of NLP models."""
tokens = tokenize(corpus)

word_counts = Counter(tokens)
vocab = {w: i for i, (w, _) in enumerate(word_counts.items())}
id2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

def generate_pairs(tokens, window_size=2):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((center, [tokens[j]]))
    return pairs

pairs = generate_pairs(tokens, window_size=2)
print("Sample Training pairs:", pairs[:10])

training_data = [(vocab[c], vocab[o[0]]) for c, o in pairs]

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words):
        embed = self.embeddings(center_words)
        out = self.linear(embed)
        return out

embed_dim = 50
model = Word2Vec(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    total_loss = 0
    for center, context in training_data:
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)

        optimizer.zero_grad()
        output = model(center_tensor)
        loss = criterion(output, context_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/50], Loss: {total_loss:.4f}')

embeddings = model.embeddings.weight.data
print("\nWord Embedding for 'language':\n", embeddings[vocab['language']])
print("\nWord Embedding for 'deep':\n", embeddings[vocab['deep']])

