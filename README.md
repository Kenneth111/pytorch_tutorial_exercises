# pytorch_tutorial_exercises
Python codes for exercises in the official Pytorch tutorials

## sequence_models_tutorial.py
[SEQUENCE MODELS AND LONG-SHORT TERM MEMORY NETWORKS](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py)

This program is for the exercise: Augmenting the LSTM part-of-speech tagger with character-level features.

## word_embeddings_tutorial.py
[WORD EMBEDDINGS: ENCODING LEXICAL SEMANTICS](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

This program is for the exercise: Computing Word Embeddings: Continuous Bag-of-Words.

## seq2seq_simple_decoder.py
[TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

This program demonstrates how to train a seq2seq model without attention mechanism.

## seq2seq_translation_batch_training.py & seq2seq_translation_batch_training.ipynb
[TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

The original program shown in the above link assumes the batch size is one. Apparently, this is not a "real" batch training. The seq2seq_translation_batch_training.py & seq2seq_translation_batch_training.ipynb show how to use batch training.

## finetuning_torchvision_models_resnet50.ipynb

[FINETUNING TORCHVISION MODELS](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

At the end of this tutorial, the author asked readers to run the code with a harder dataset. I run Resnet with [the plant seeding classification dataset](https://www.kaggle.com/c/plant-seedlings-classification). In my code, I show how to add multiple layers to the top of a deep neural network model and how to use pretrained models in a Kaggle kernel.

## ImageFolderSplitter.py

Two classes, ImageFolderSplitter and DatasetFromFilename, are provided in this file. They work like torchvision.datasets.ImageFolder, but they can split a whole dataset into a training set and a validation set.