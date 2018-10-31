# -*- coding: utf-8 -*-
# the following codes are for the exercise in https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Prepare data
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 5

# generate a dict like: {"a": 0, "b": 1, ..., "z": 25}
char2ix = {chr(i): i - 97 for i in range(97, 97 + 26)}

# convert a word to a tensor
def char2seq(word, char2ix):
    char_seq = [char2ix[c.lower()] for c in word]
    return torch.tensor(char_seq, dtype=torch.long)

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_word_dim, hidden_char_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        # now we need two hidden state dimensions, because there're two lstm layers.
        self.hidden_word_dim = hidden_word_dim
        self.hidden_char_dim = hidden_char_dim
        # word and character embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings = nn.Embedding(26, embedding_dim)
        # lstm layers for processing character embeddings and the combination of 
        # character level representations and word embeddings
        self.char_lstm = nn.LSTM(embedding_dim, hidden_char_dim)
        self.word_lstm = nn.LSTM(embedding_dim + hidden_char_dim, hidden_word_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_word_dim, tagset_size)
        # initialize hidden states
        self.hidden_word = self.init_hidden(self.hidden_word_dim)
        self.hidden_char = self.init_hidden(self.hidden_char_dim)

    def init_hidden(self, dim):
        return (torch.zeros(1, 1, dim),
                torch.zeros(1, 1, dim))

    def forward(self, sentence, words):
        # generate character representations
        char_list = []
        for word in words:
            # clear out the hidden state
            self.hidden_char = self.init_hidden(self.hidden_char_dim)
            char_embeds = self.char_embeddings(word)
            _, self.hidden_char = self.char_lstm(char_embeds.view(len(word), 1, -1))
            char_list.append(self.hidden_char[0])
        char_list = torch.stack(char_list).view(len(words), -1)

        word_embeds = self.word_embeddings(sentence)
        # merge character representations with word embeddings
        cat_embeds = torch.cat([word_embeds, char_list], dim=1)
        lstm_out, self.hidden_word = self.word_lstm(
            cat_embeds.view(len(sentence), 1, -1), self.hidden_word)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# training
tagger_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 3, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(tagger_model.parameters(), lr=0.1)
for epoch in range(200):
    for sentence, tags in training_data:
        tagger_model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tagger_model.hidden_word = tagger_model.init_hidden(HIDDEN_DIM)
        char_seq = [char2seq(word, char2ix) for word in sentence]
        tag_scores = tagger_model(sentence_in, char_seq)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# testing
# create a dict, {0: 'DET', 1: 'NN', 2: 'V'}
ix2tag = {tag_to_ix[tag]: tag for tag in tag_to_ix.keys()}
with torch.no_grad():
    for i in range(2):
        sentence = training_data[i][0]
        print(sentence)
        inputs = prepare_sequence(sentence, word_to_ix)
        char_seq = [char2seq(word, char2ix) for word in sentence]
        tag_scores = tagger_model(inputs, char_seq)
        # find the index of the max value in each row
        tags = torch.argmax(tag_scores, dim=1).numpy()
        print([ix2tag[idx] for idx in tags])