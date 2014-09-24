# Problem Set 2 #

This pset is about sequence labeling, focusing on part-of-speech tagging for Twitter. You will implement sequence labeling algorithms that are near the state-of-the-art for this task. There are two parts:

- [Problem set 2a](ps-2a.ipynb), which includes a most-common-class baseline, classification-based tagging using Naive Bayes, the Viterbi algorithm, and Hidden Markov Models (HMM). It is assigned on September 16 and due on September 23.
- [Problem set 2b](ps-2b.ipynb), which includes classification-based tagging with averaged perceptron, the design of useful features for tagging, and structured perceptron for discriminative sequence labeling. It is assigned on September 24 and due on October 1 at 6pm. As in pset 1, there will be a bakeoff component, and you will get unlabeled test data shortly before the due date.

You will need the following files:
- [training data](oct27.train?raw=true)
- [development data](oct27.dev?raw=true)
- [test data](oct27.test?raw=true)
- [scorer code](scorer.py?raw=true)

You can get all of this by cloning the entire course repository using
"git clone https://github.com/jacobeisenstein/gt-nlp-class.git".
If you've already done this, just update the ps1 directory.

## Submission and honor policy ##
On T-square, you should submit your version of the notebook. All code should be in the notebook. In Part 2b, please also submit your response files to the test data using the appropriate filenames (described in the last section of the notebook).

If you have questions about the assignment, please use the class
[email list](https://groups.google.com/forum/#!forum/gt-nlp-class-fa2014),
and not a personal email to the class staff.

Your work should be your own, so please do not discuss the details of
the assignment. You may of course help each other with understanding the ideas discussed in lecture and the readings, and with basic questions about programming in Python. There are implementations and source code for many machine learning algorithms on the internet. Please write the code for this assignment on your own, without using these external resources.
