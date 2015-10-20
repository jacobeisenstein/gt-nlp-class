CS 4650 and CS 7650 will meet jointly, on Mondays and Wednesdays from 3:05 - 4:25PM, in Howey (Physics) L3. Office hours are listed [here](./Policies.md)

This is a provisional schedule. Check back in August for more details. But be aware that **there will be graded material due no later than the second week of the course (maybe even the first week).**

Readings and homeworks are final at the time of the class **before** they are due (e.g., wednesday readings are final on the preceding monday); problem sets are final on the day they are "out." Please check for updates until then.

### August 17: Welcome ###

History of NLP and modern applications. Review of probability. 

- **Reading**: Chapter 1 of [Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020).
You should be able to access this PDF for free from a Georgia Tech computer.
- **Optional reading**: [Functional programming in Python](https://docs.python.org/2/howto/functional.html).
The scaffolding code in this class will make heavy use of Python's functional programming features, such as iterators, generators, list comprehensions, and lambda expressions. If you haven't seen much of this style of programming before, it will be helpful for you to read up on it before getting started with the problem sets.
- **Optional reading**: Section 2.1 of [Foundations of Statistical NLP](http://nlp.stanford.edu/fsnlp/). A PDF version is accessible through the GT library.
- **Optional reading** includes [these](http://www.autonlab.org/tutorials/prob18.pdf) 
[other](http://www.cs.cmu.edu/~tom/10701_sp11/slides/Overfitting_ProbReview-1-13-2011-ann.pdf) [reviews](http://www.cs.cmu.edu/~tom/10701_sp11/slides/MLE_MAP_1-18-11-ann.pdf)
of probability.
- **[Why you should take notes by hand, not on a laptop](http://www.vox.com/2014/6/4/5776804/note-taking-by-hand-versus-laptop)**
- **[Problem set 1](psets/ps1.pdf)** out.

### August 19: Supervised learning 1 (Naive Bayes) ###

Bag-of-words models, naive Bayes, and sentiment analysis.

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 1, 3.1.
- Optional readings:
[Sentiment analysis and opinion mining](http://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html), especially parts 1, 2, 4.1-4.3, and 7;
[Chapters 0-0.3, 1-1.2 of LXMLS lab guide](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/lxmls-guide.pdf?raw=true)
- [Homework 1](homeworks/homework-1.md) cancelled, since waitlist students can't yet access the course t-square. We'll try this again, and the total number of homeworks will remain 12.
- [Demo](classes/Lec-2 Simple Sentiment Analysis.ipynb)

### August 24: Supervised learning 2 (Perceptron) ###

Discriminative classifiers: perceptron and passive-aggressive learning; word-sense disambiguation. 

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 2-2.3.
- Optional supplementary reading: Parts 4-7 of [log-linear models](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-loglin.pdf?raw=true); [survey on word sense disambiguation](http://promethee.philo.ulg.ac.be/engdep1/download/bacIII/ACM_Survey_2009_Navigli.pdf)
- Optional advanced reading: 
[adagrad](jmlr.org/papers/v12/duchi11a.html); [passive-aggressive learning](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)

### August 26: Supervised learning 3 (Logistic regression) ###

Logistic regression and online learning

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 2.4-2.6.
- Optional supplementary reading: Parts 4-7 of [log-linear models](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-loglin.pdf?raw=true)
- **[Problem set 1](psets/ps1.pdf)** due at 2:55pm.
- **[Problem set 2](psets/pset2.md)** out on August 28

### August 31: Review of classifiers and Word Sense Disambiguation ###

- Reading: notes chapter 3.2
- [Homework 2](homeworks/homework-2.md) due

### September 2: Expectation Maximization ###

Learning from partially-labeled data.

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 4-4.3.
- Online [demo](demos/EM-demo.ipynb)

### September 7: Official school holiday ####

[Labor Day](https://en.wikipedia.org/wiki/Labor_Day) is a celebration of the American Labor Movement.

### September 9: Language Models ###

N-grams, speech recognition, smoothing, recurrent neural networks.

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 5.
- [Homework 3](homeworks/homework-3.md) due
- [Demo](demos/lm.ipynb)
- Optional advanced reading: [An empirical study of smoothing techniques for language models](http://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf), especially sections 2.7 and 3 on Kneser-Ney smoothing; [A hierarchical Bayesian language model based on Pitman-Yor processes](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf) (requires some machine learning background)

### September 14: Morphology, Stemming, and Lemmatisation  ###

Finding meaning inside words! Also, we'll probably have to catch up a little on smoothing from the previous class.

- **[Homework 4](homeworks/homework-4.md)** due
- Reading: [Bender](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020) chapter 2.
- Optional reading:
[Jurafsky and Martin](http://books.google.com/books/about/Speech_and_language_processing.html?id=km-kngEACAAJ) chapter 2.

### September 16: Finite-state automata ###

Finite-state acceptors, transducers, composition. Edit distance.

- **[Problem set 2](psets/pset2.md)** due at 2:55 pm.
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 7
- Optional reading: [Knight and May](http://ai.eecs.umich.edu/people/rounds/csli/main.pdf);
[OpenFST slides](http://www.stringology.org/event/CIAA2007/pres/Tue2/Riley.pdf);
[Weighted Finite-State Transducers in speech recognition](http://www.cs.nyu.edu/~mohri/pub/csl01.pdf).

### September 21: No class ###

- Jacob will be at the conference on [Empirical Methods in Natural Language Processing](http://www.emnlp2015.org/), presenting research from the [Computational Linguistics Lab](https://gtnlp.wordpress.com/).

### September 23: Part-of-speech tagging and Hidden Markov Models ###

Part-of-speech tags, hidden Markov models.

- [Homework 5](homeworks/homework-5.md) due
- **[Problem set 3](psets/pset3.md)** out.
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 8
- Optional reading: Bender chapter 6; [Tagging problems and hidden Markov models](http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf)

### September 28: Dynamic Programming in Hidden Markov Models ###

Viterbi, the forward algorithm, B-I-O encoding for named entity recognition.

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 9-9.5
- Optional reading: [Conditional random fields](http://www.cs.columbia.edu/~mcollins/crf.pdf);
- [Slides](classes/sequence-slides.pdf)

### September 30: No class ###

- TAs will be available to answer questions on problem set 3.
- Jacob will be presenting research at [DiSpoL 2015](http://www.coli.uni-saarland.de/conf/dispol2015/), a workshop on discourse structure.

### October 5: Discriminative Sequence Labeling ###

Structured perceptron, conditional random fields, and max-margin markov networks. More about forward-backward. Maybe a little about unsupervised POS tagging.

- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 9.6-9.9 
- Optional reading: [Discriminative training of HMMs](http://dl.acm.org/citation.cfm?id=1118694); [CRF tutorial](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf);
[Two decades of unsupervised POS tagging: how far have we come?](homepages.inf.ed.ac.uk/sgwater/papers/emnlp10-20yrsPOS.pdf); my notes 9.10
- **[Problem set 3](psets/pset3.md)** due at 2:55pm.

### October 7: Context-Free Grammars and Natural Language Syntax ###

Constituents, grammar design, formal language theory.

- Reading: my notes, chapter 10
- Optional reading: Bender chapter 7
- **[Problem set 4](psets/pset4.md)** out.

### October 12: No class, fall break ###

### October 14: CFG Parsing ###

The CKY algorithm, the inside algorithm, Markovization, and lexicalization.

- [Homework 6](homeworks/homework-6.md) due
- Reading: my notes, chapter 10.4-11.2
- Optional reading: [Probabilistic context-free grammars](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf); Bender chapter 8; my notes 10.13-10.14. 

### October 19: Mid-term exam ###

You may bring a one-page sheet of notes (two sides, any font size).

### October 21: Statistical Parsing ###

Brief mid-term review. Constituency parsing: Markovization, lexicalization, refinement grammars. Intro to dependency parsing.

- Reading: my notes, chapter 11.3-12
- Optional reading: [Eisner algorithm worksheet](http://www.cc.gatech.edu/~jeisenst/classes/cs7650_sp12/eisner_worksheet.pdf);
[Characterizing the errors of data-driven dependency parsing models](http://acl.ldc.upenn.edu/D/D07/D07-1013.pdf);
[Short textbook on dependency parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002), PDF should be free from a GT computer.
- The always useful [language log](http://languagelog.ldc.upenn.edu/nll/?p=7851) on non-projectivity in dependency parsing.

### October 25: Withdrawal Deadline ###

### October 26: More Dependency Parsing, and Mildly Context-Sensitive Grammars ###

Dependency grammar, projective and non-projective dependency graphs, related algorithms, and transition-based dependency parsing. Quick tour of feature-structure grammars, unification, combinatory categorial grammar (CCG), tree-adjoining grammar (TAG). Algorithms and applications.

- [Homework 7](homeworks/homework-8.md) due
- Reading: Finish chapter 12 of my notes; [intro to CCG](readings/ccgintro.pdf?raw=true);
- Optional reading: [The inside-outside algorithm](http://www.cs.columbia.edu/~mcollins/io.pdf); 
[Corpus-based induction of linguistic structure](http://acl.ldc.upenn.edu/acl2004/main/pdf/341_pdf_2-col.pdf);
[Much more about CCG](http://homepages.inf.ed.ac.uk/steedman/papers/ccg/SteedmanBaldridgeNTSyntax.pdf); [LTAG](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog2805_2/pdf); [Probabilistic disambiguation models for wide-coverage HPSG](http://acl.ldc.upenn.edu/P/P05/P05-1011.pdf)
- **[Problem set 4](psets/pset4.md)** due at 2:55pm.

### October 28: Formal Semantics ###

Meaning representations, compositionality, first-order logic, and the syntax-semantics interface.

- Reading: my notes, chapter 13 (if ready)
- Optional readings: [Levy and Manning: Intro to Formal Computational Semantics](findit);
[Learning to map sentences to logical form](http://arxiv.org/pdf/1207.1420v1.pdf); 
- **[Problem set 5](psets/pset5.md)** out.

### November 2: Shallow Semantics ###

PropBank, FrameNet, semantic role labeling, and a little Abstract Meaning Representation (AMR). Integer linear programming will also be discussed.

- [Homework 8](homeworks/homework-9.md) due
- Reading: my notes chapter 14 (if ready). Else: [Gildea and Jurafsky](http://web.stanford.edu/~jurafsky/cl01.pdf)  sections 1-3; [Banarescu et al](http://amr.isi.edu/a.pdf) sections 1-4
- Optional reading:  [SRL via ILP](https://www.aclweb.org/anthology/C/C04/C04-1197.pdf); [Syntactic parsing in SRL](http://www.aclweb.org/anthology/J/J08/J08-2005.pdf);
[AMR parsing](http://www.cs.cmu.edu/~jmflanig/flanigan+etal.acl2014.pdf)
- Optional [video](http://videolectures.net/metaforum2012_pereira_semantic/)

### November 4: Anaphora and Coreference Resolution ###

Classification-based algorithms; graph-based algorithms; a brief intro to government and binding theory.

- [Homework 9](homeworks/homework-9.md) due
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 17
- Optional reading: [Multi-pass sieve](http://www.surdeanu.info/mihai/papers/emnlp10.pdf)  (good coverage of linguistic features that bear on coreference); [Large-scale multi-document coreference](http://people.cs.umass.edu/~sameer/files/largescale-acl11.pdf), [Easy victories and uphill battles](http://www.eecs.berkeley.edu/~gdurrett/papers/durrett-klein-emnlp2013.pdf) (a straightforward machine learning approach to coreference)

### November 9: Discourse and Dialogue ###

Coherence, cohesion, centering theory, topic segmentation, speech act classification.

- Reading: [Discourse structure and language technology](http://journals.cambridge.org/repo_A84ql5gR)
- [Homework 10](homeworks/homework-10.md) due
- Optional:
[Modeling local coherence](http://www.aclweb.org/anthology-new/J/J08/J08-1001.pdf); [Sentence-level discourse parsing](http://acl.ldc.upenn.edu/N/N03/N03-1030.pdf); [Analysis of discourse structure...](http://people.ict.usc.edu/~sagae/docs/sagae-discourse-iwpt09.pdf)
- **[Problem set 5](psets/pset5.md)** due at 2:55pm.

### November 11: Lexical and Distributional Semantics ###

Latent semantic analysis, word embeddings

- Reading: [Vector-space models](https://www.jair.org/media/2934/live-2934-4846-jair.pdf), sections 1, 2, 4-4.4, 6
- Optional:  my notes, chapter 15
- **[Problem set 6](psets/pset6.md)** out.
- Optional reading: [python coding tutorial](http://radimrehurek.com/2014/02/word2vec-tutorial/) for word2vec word embeddings

### November 16: Machine Translation ###

- [Homework 11](homeworks/homework-11.md) due
- Reading: Collins, [IBM models 1 and 2](papers/collins-ibm12.pdf)
- Optional Reading: Chiang, [Intro to Synchronous Grammars](http://www.isi.edu/~chiang/papers/synchtut.pdf);
Lopez, [Statistical machine translation](http://www.cs.jhu.edu/~alopez/papers/survey.pdf)

### November 18: Information Extraction ###

Reading for comprehension.

- [Homework 12](homeworks/homework-12.md) due
- Reading: [Grishman](http://cs.nyu.edu/grishman/tarragona.pdf), sections 1 and 4-6
- Optional reading: there's lots, TBD

### November 23: Alternative Training Scenarios for NLP ###

Semi-supervised learning and domain adaptation.

- **[Problem set 6](psets/pset6.md)** due at 2:55pm.
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 19
- Optional reading: [Jerry Zhu's survey](http://pages.cs.wisc.edu/~jerryzhu/pub/SSL_EoML.pdf);
[Jerry Zhu's book](http://www.morganclaypool.com/doi/abs/10.2200/S00196ED1V01Y200906AIM006)

### November 25: Thanksgiving ###

No class.

### November 30: Slush day ###

We'll use this class to make up on time, in case we are behind.
If by some miracle we are not behind, we can talk about the **best papers** from [NAACL](http://naacl.org/naacl-hlt-2015/best-paper-awards.html), [ACL](http://acl2015.org/abstracts_long/91.html), and [EMNLP](http://www.emnlp2015.org/best-papers.html) 2015. 

### December 2: Exam Review ###

- Reading: TBD
- [Homework 13](homeworks/homework-12.md) due

### December 9: Final Exam ###

*2:50 - 5:40pm.* You may bring a single sheet of notes, two-sided.
