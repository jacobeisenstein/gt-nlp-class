Homework 3: A Poet's Guide to Unix
================

This homework is about doing first-pass text analytics on the
command-line. The ultimate guide to this was written by Ken Church,
and can be downloaded from
[http://ufal.mff.cuni.cz/~hladka/tutorial/UnixforPoets.pdf](http://ufal.mff.cuni.cz/~hladka/tutorial/UnixforPoets.pdf).

The guide uses the book of Genesis as a source of examples; for this homework, use the Project Gutenberg text
of [Moby Dick](http://www.gutenberg.org/cache/epub/2701/pg2701.txt), a
book about an angry whale. (Click through to ```Plain Text UTF-8``` to download.)

Follow the instructions at least as far as Section 4, where you compute bigram counts.
Some of the instructions are a little out of date: for example, ```tail +2``` doesn't work
for me, but ```tail -n +2``` accomplishes the same thing.

# Deliverable 1 #

When we were discussing bag-of-words models, several students raised
multiword phenomena as a problem. A potential solution is to
identify [collocations](https://en.wikipedia.org/wiki/Collocation),
which are pairs of words that tend to go together; we can then treat
collocations as special terms. One way to identify collocations is by
[pointwise mutual information (PMI)](https://en.wikipedia.org/wiki/Pointwise_mutual_information#Normalized_pointwise_mutual_information_.28npmi.29).

Compute the pointwise mutual information of the bigram "Captain Ahab"
(case sensitive). Specifically, let $N$ equal the total word count of
the document (obtained from ```wc```), and compute,

$P("Captain","Ahab") = \frac{\text{count}(x_i = \text{"Captain"},
x_{i+1} = \text{"Ahab"})}{N}$

Then you can compute the PMI as 

$PMI("Captain Ahab") = \log \frac{P("Captain","Ahab")}{P("Captain")P("Ahab")}.$

Please list the intermediate values that you compute, e.g. the relevant unigram and bigram counts, etc.

# Deliverable 2 #

Compute the PMI for the same bigram, this time case insensitive.

# Deliverable 3 #

Identify two more collocations of interest, for which the bigram count is at least 10.

