# Problem Set 2: Sequence Labeling

There are two notebook files here:

- `pytorch_tensors.ipynb`: This notebook briefly describes some basic operations on Tensors in PyTorch. All these will be very useful in this problem set. So, make sure to go through them before starting the problem set.

- `pset2.ipynb`: This notebook is the problem set. This problem set focuses on sequence labeling with Hidden Markov models and Deep Learning Models. The target domain is part-of-speech tagging on English and Norwegian from the Universal Dependencies dataset.
    You will:
    - Do some basic preprocessing of the data
    - Build a naive classifier that tags each word with its most common tag
    - Implement a `Viterbi` Tagger using `Hidden Markov Model` in PyTorch
    - Build a `Bi-LSTM` deep learning model using PyTorch
    - Build a `Bi-LSTM_CRF` model using the above components (`Viterbi` and `Bi-LSTM`) 
    - then implement techniques to improve your classifier and compete on Kaggle.
