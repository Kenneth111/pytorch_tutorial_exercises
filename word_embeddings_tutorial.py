# -*- coding: utf-8 -*-
# for the exercise in https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

EMBEDDING_DIM = 10
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        # sum the embeddings of the context
        x = torch.sum(x, dim=0)
        x = x.view((1, -1))
        x = self.linear(x)
        return F.log_softmax(x)

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


model = CBOW(vocab_size, EMBEDDING_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# training
for i in range(200):
    for context, target in data:
        model.zero_grad()
        predict = model(make_context_vector(context, word_to_ix))
        loss = loss_function(predict, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

# testing
ix2word = {word_to_ix[key]: key for key in word_to_ix.keys()}
for context, target in data:
    print(context, target)
    predict = model(make_context_vector(context, word_to_ix))
    max_idx = torch.argmax(predict, dim=1).numpy()
    print(ix2word[max_idx[0]])