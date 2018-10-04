# Tweet encoders
author: Peter Frick

Word2vec is an established algorithm to learn vector embeddings for individual words
or items through a shallow network. Learning embeddings at the document level (e.g., collection
of words) is a more recent area of research.

This repo is meant to explore some different network architectures for getting embeddings at the
tweet level. They are as follows:

1. maxpool encoder
2. prediction autoencoder
3. embeddings autoencoder

Each are explored in an individual `*.ipynb` notebooks.

Pretrained glove embeddings for tweets are publically available [here](https://nlp.stanford.edu/projects/glove/)


