# Problem Set 1 #

This pset is about sentiment classification, and you will write code for running and evaluating various classifiers. There are two parts:

- [Problem set 1a](ps-1a.ipynb), which includes dictionary-based classification and Naive Bayes, and goes out on August 26, due September 3. To start, open the linked document in iPython Notebook.
- [Problem set 1b](ps-1b.ipynb), which includes perceptron (and maybe something else), and goes out on September 3, due September 10. The iPython notebook will be available on September 3.

You'll also need this data:
- [training data](train-imdb.tgz?raw=true)
- [training key](train-imdb.key?raw=true)
- [development data](dev-imdb.tgz?raw=true)
- [development key](dev-imdb.key?raw=true)
- [sentiment vocabulary](sentiment-vocab.tff?raw=true) by Wiebe et al
- [scorer code](scorer.py?raw=true)

You can get all of this by cloning the entire course repository using
"git clone https://github.com/jacobeisenstein/gt-nlp-class.git".
If you've already done this, just update the ps1 directory.

## Ipython notebook ##
This problem set involves
[ipython notebook](http://ipython.org/notebook.html), a web-based IDE
for python. Notebook allows us to provide you a code scaffolding, and
allows you to annotate your code with comments in
[Markdown](http://en.wikipedia.org/wiki/Markdown) and
[LaTeX](http://en.wikipedia.org/wiki/LaTeX). Please take some time to
familiarize yourself with this coding environment. I'm currently using
the [2.2.0 version] of the notebook, with Python 2.7.5.

## Some resources: ##

- [a numerical python cheatsheet](http://mathesaurus.sourceforge.net/matlab-python-xref.pdf)
  for matlab and R users
- [NLTK](http://nltk.org/), the Natural Language Toolkit, in Python. Note that [the book](http://nltk.org/book/) is available for free.
- [Scikit-learn](http://scikit-learn.org/stable/), a machine learning library in python.

## Submission and honor policy ##

On T-square, you should submit your version of the notebook. All code should be in the notebook. In Part 1b, please also submit your response files to the test data using the appropriate filenames (described in the last section of the notebook).

If you have questions about the assignment, please use the class
[email list](https://groups.google.com/forum/#!forum/gt-nlp-class-fa2013),
and not a personal email to the class staff.

Your work should be your own, so please do not discuss the details of
the assignment. You may of course help each other with understanding the ideas discussed in lecture and the readings, and with basic questions about programming in Python. There are implementations and source code for many machine learning algorithms on the internet. Please write the code for this assignment on your own, without using these external resources.
