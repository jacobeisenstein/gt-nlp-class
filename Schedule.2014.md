CS 4650 and CS 7650 will meet jointly, on Tuesdays and Thursdays from 3:05 - 4:25PM, in College of Computing 101.

**This is a (permanently) provisional schedule.** Readings, notes, slides, and homework will change. Readings and homeworks are final at the time of the class **before** they are due (e.g., thursdays readings are final on the preceding tuesday); problem sets are final on the day they are "out." Please check for updates until then.


### August 19: Welcome ###

- History of NLP and modern applications. Review of probability. 
- **Reading**: Chapter 1 of [Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020).
You should be able to access this PDF for free from a Georgia Tech computer.
- **Optional reading**: [Functional programming in Python](https://docs.python.org/2/howto/functional.html).
The scaffolding code in this class will make heavy use of Python's functional programming features, such as iterators, generators, list comprehensions, and lambda expressions. If you haven't seen much of this style of programming before, it will be helpful for you to read up on it before getting started with the problem sets.
- **Optional reading**: Section 2.1 of [Foundations of Statistical NLP](http://nlp.stanford.edu/fsnlp/). A PDF version is accessible through the GT library.
- **Optional reading** includes [these](http://www.autonlab.org/tutorials/prob18.pdf) 
[other](http://www.cs.cmu.edu/~tom/10701_sp11/slides/Overfitting_ProbReview-1-13-2011-ann.pdf) [reviews](http://www.cs.cmu.edu/~tom/10701_sp11/slides/MLE_MAP_1-18-11-ann.pdf)
of probability.
- [Project 0](psets/ps0.pdf?raw=true) out
- [Slides](slides/lec1.pdf?raw=true)

### August 21: Supervised learning 1 (Naive Bayes) and sentiment analysis ###
- Bag-of-words models, naive Bayes, and sentiment analysis.
- [Homework 1](homeworks/homework-1.md) due
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 3.
- Optional readings:
[Sentiment analysis and opinion mining](http://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html), especially parts 1, 2, 4.1-4.3, and 7;
[Chapters 0-0.3, 1-1.2 of LXMLS lab guide](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/lxmls-guide.pdf?raw=true)
- [Slides](slides/lec2-slides.pdf?raw=true)

### August 26: Supervised learning 2 (Perceptron) and word sense disambiguation ###
- Discriminative classifiers: perceptron and passive-aggressive learning; word-sense disambiguation. 
- [Problem set 0](psets/ps0.pdf?raw=true) due
- [Problem set 1a](psets/ps1/ps1.md) out
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 5-5.2.
- Optional supplementary reading: Parts 4-7 of [log-linear models](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-loglin.pdf?raw=true); [survey on word sense disambiguation](http://promethee.philo.ulg.ac.be/engdep1/download/bacIII/ACM_Survey_2009_Navigli.pdf)
- Optional advanced reading: 
[adagrad](jmlr.org/papers/v12/duchi11a.html); [passive-aggressive learning](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
- [Slides](slides/lec3-slides.pdf?raw=true)

### August 28: Supervised learning 3 (Logistic regression) ###

- Logistic regression and online learning
- [Homework 2](homeworks/homework-2.md) due
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 5.3-5.6.
- Optional supplementary reading: Parts 4-7 of [log-linear models](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-loglin.pdf?raw=true)
- [Slides](slides/lec4-slides.pdf?raw=true)

### September 2: Expectation maximization and semi-supervised learning; language models ###

- [Problem set 1a](psets/ps1/ps1.md) due on **September 3 at 3pm**
- [Problem set 1b](psets/ps1/ps1.md) out on **September 3 at 3pm**
- Reading: [Expectation maximization](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-em.pdf?raw=true)
chapter by Michael Collins
- Optional supplementary reading: [Tutorial on EM](http://www.cc.gatech.edu/~dellaert/em-paper.pdf)
- Optional advanced reading: [Nigam et al](http://www.kamalnigam.com/papers/emcat-mlj99.pdf); [Word](http://acl.ldc.upenn.edu/P/P95/P95-1026.pdf)  [sense](http://www.d.umn.edu/~tpederse/Pubs/wsdbook-2006-pedersen.pdf) [clustering](http://www.aclweb.org/anthology-new/W/W97/W97-0322.pdf)
- Demo: [Word sense clustering with EM](demos/word-cluster.ipynb)
- [Slides](slides/lec5-slides.pdf?raw=true)

### September 4: Language models, smoothing, and speech recognition ###

- N-grams, smoothing, speech recognition
- Reading: [Language modeling](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-lm.pdf?raw=true) 
- [Homework 3](homeworks/homework-3.md) due
- Optional advanced reading: [An empirical study of smoothing techniques for language models](http://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf), especially sections 2.7 and 3 on Kneser-Ney smoothing; [A hierarchical Bayesian language model based on Pitman-Yor processes](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf) (requires some machine learning background)
- [Slides](slides/lec6-slides.pdf?raw=true)
- [Demo](demos/lm.ipynb)

### September 9: Finite state automata, morphology, semirings ###

- [Problem set 1b](psets/ps1/ps1.md) due on **September 10 at 3pm**
- Reading: [Knight and May](http://ai.eecs.umich.edu/people/rounds/csli/main.pdf) (section 1-3)
- Supplemental reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 10-10.3; [Jurafsky and Martin](http://books.google.com/books/about/Speech_and_language_processing.html?id=km-kngEACAAJ) chapter 2.
- [Slides](slides/lec7-morphology-slides.pdf?raw=true) on morphology

### September 11: Finite state transducers ###

- Transduction and composition, edit distance
- [Homework 4](homeworks/homework-4.md) due
- Reading: Chapter 2 of [Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020)
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 10.4- (not done yet) 
- Optional reading: [OpenFST slides](http://www.stringology.org/event/CIAA2007/pres/Tue2/Riley.pdf).
- More formal additional reading: [Weighted Finite-State Transducers in speech recognition](http://www.cs.nyu.edu/~mohri/pub/csl01.pdf)
- [Slides](slides/lec8-slides.pdf?raw=true)

### September 16: Sequence labeling 1 ###

- Part-of-speech tags, hidden Markov models. 
- [Problem set 2a](psets/ps2/ps2.md) out
- Reading: Bender chapter 6
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapters 11 and 12.
- Optional reading: [Tagging problems and hidden Markov models](http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf)
- [Slides](slides/sequence-slides.pdf?raw=true)

### September 18: Sequence labeling 2 ###

- Viterbi, the Forward algorithm, and B-I-O encoding. 
- [Homework 5](homeworks/homework-5.md) due
- Reading: [Conditional random fields](http://www.cs.columbia.edu/~mcollins/crf.pdf)
- Optional reading: [CRF tutorial](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf); [Discriminative training of HMMs](http://dl.acm.org/citation.cfm?id=1118694)

### September 23: Sequence labeling 3 ###

- Discriminative structure prediction, conditional random fields, and the forward-backward algorithm.
- [Problem set 2a](psets/ps2/ps2.md) due
- [Problem set 2b](psets/ps2/ps2.md) out (September 24)
- Reading: [Forward-backward](http://www.cs.columbia.edu/~mcollins/fb.pdf)
- Optional reading: [Two decades of unsupervised POS tagging: how far have we come?](homepages.inf.ed.ac.uk/sgwater/papers/emnlp10-20yrsPOS.pdf)

### September 25: Syntax and CFG parsing ###

- Context-free grammars; constituency; parsing
- [Homework 6](homeworks/homework-6.md) due
- Reading: [Probabilistic context-free grammars](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf) 
- Optional reading: My notes, chapter 13.
- [Slides on parsing](slides/parsing-slides.pdf?raw=true)

### September 30: Dependency parsing ###

- [Problem set 2b](psets/ps2/ps2.md) due (October 1, 5pm)
- Reading: my notes, chapter 14.
- Optional reading: [Eisner algorithm worksheet](http://www.cc.gatech.edu/~jeisenst/classes/cs7650_sp12/eisner_worksheet.pdf);
[Characterizing the errors of data-driven dependency parsing models](http://acl.ldc.upenn.edu/D/D07/D07-1013.pdf);
[Short textbook on dependency parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002), PDF should be free from a GT computer.
- [Slides on dependency parsing](slides/lec14-depparsing-slides.pdf?raw=true)
- The always useful [language log](http://languagelog.ldc.upenn.edu/nll/?p=7851) on non-projectivity in dependency parsing.

### October 2: Catch-up, midterm review ###

- [Homework 7](homeworks/homework-7.md) due

### October 7: Midterm ###

- [Minimal review notes](notes/review.pdf?raw=true)

### October 9: Midterm recap, modern parsing ###
- Reading: [Lexicalized PCFGs](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/lexpcfgs.pdf)
- Reading: my notes, sections 13.13 and 13.14
- [Slides](slides/modern-cfg-parsing.pdf?raw=true)
- Optional reading: [Accurate unlexicalized parsing](http://acl.ldc.upenn.edu/P/P03/P03-1054.pdf)

### October 10: Drop deadline ###

### October 14: Fall recess, no class ###

- [Problem set 3](psets/ps3) out

### October 16: Alternative models of syntax  ###

- Mostly CCG, but a little about L-TAG and and HPSG.
- [Homework 8](homeworks/homework-8.md) due
- Reading: [Intro to CCG](readings/ccgintro.pdf?raw=true);
- [Slides](slides/beyond-cfg.pdf?raw=true)
- Optional reading: [The inside-outside algorithm](http://www.cs.columbia.edu/~mcollins/io.pdf); 
[Corpus-based induction of linguistic structure](http://acl.ldc.upenn.edu/acl2004/main/pdf/341_pdf_2-col.pdf);
[Much more about CCG](http://homepages.inf.ed.ac.uk/steedman/papers/ccg/SteedmanBaldridgeNTSyntax.pdf); [LTAG](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog2805_2/pdf); [Probabilistic disambiguation models for wide-coverage HPSG](http://acl.ldc.upenn.edu/P/P05/P05-1011.pdf)

### October 21: Compositional logical semantics ###

- [Homework 9](homeworks/homework-9.md) due
- Reading: [Manning: Intro to Formal Computational Semantics](http://www.stanford.edu/class/cs224u/readings/cl-semantics-new.pdf)
- Optional reading: [Learning to map sentences to logical form](http://arxiv.org/pdf/1207.1420v1.pdf); 
- [Slides](slides/formal-semantics-slides.pdf?raw=true)

### October 23: Shallow semantics ###

- Frame semantics, and semantic role labeling. 
- [Homework 10](homeworks/homework-10.md) due
- [Problem set 3](psets/ps3) due
- Reading: [Gildea and Jurafsky](http://web.stanford.edu/~jurafsky/cl01.pdf)  sections 1-3; [Banarescu et al](http://amr.isi.edu/a.pdf) sections 1-4
- Optional reading:  [SRL via ILP](https://www.aclweb.org/anthology/C/C04/C04-1197.pdf); [Syntactic parsing in SRL](http://www.aclweb.org/anthology/J/J08/J08-2005.pdf);
[AMR parsing](http://www.cs.cmu.edu/~jmflanig/flanigan+etal.acl2014.pdf)
- Optional [video](http://videolectures.net/metaforum2012_pereira_semantic/)
- [Slides](slides/srl-slides.pdf?raw=true)
- [Notes on Integer Linear Programming for SRL](notes/srl-ilp-notes.pdf?raw=true)

### October 28: Distributional semantics ###

- Vector semantics, latent semantic indexing, neural word embeddings
- [Problem set 4](psets/ps4/ps-4.md) out
- Reading: [Vector-space models](https://www.jair.org/media/2934/live-2934-4846-jair.pdf), sections 1, 2, 4-4.4, 6
- Optional: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 15;  [python coding tutorial](http://radimrehurek.com/2014/02/word2vec-tutorial/) for word2vec word embeddings
- [Slides](slides/distributional-slides.pdf?raw=true)

### October 30: Anaphora and coreference resolution ###

- Knowing who's on first.
- [Homework 11](homeworks/homework-11.md) due
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 16; [Multi-pass sieve](http://www.surdeanu.info/mihai/papers/emnlp10.pdf)  (good coverage of linguistic features that bear on coreference)
- Optional reading: [Large-scale multi-document coreference](http://people.cs.umass.edu/~sameer/files/largescale-acl11.pdf), [Easy victories and uphill battles](http://www.eecs.berkeley.edu/~gdurrett/papers/durrett-klein-emnlp2013.pdf) (a straightforward machine learning approach to coreference)
- [Slides](slides/coref-slides.pdf)

### November 4: Discourse and dialogue ###

- Coherence; speech acts, discourse connectives
- [Homework 12](homeworks/homework-12.md) due
- Reading: [Discourse structure and language technology](http://journals.cambridge.org/repo_A84ql5gR)
- Optional:
[Modeling local coherence](http://www.aclweb.org/anthology-new/J/J08/J08-1001.pdf); [Sentence-level discourse parsing](http://acl.ldc.upenn.edu/N/N03/N03-1030.pdf)
- [Slides](slides/discourse-slides.pdf)

### November 6: Discourse parsing ###

- Rhetorical structure theory, Penn Discourse Treebank
- Reading: [Analysis of discourse structure...](http://people.ict.usc.edu/~sagae/docs/sagae-discourse-iwpt09.pdf)
- [Problem set 4](psets/ps4/ps-4.md) due

### November 11: Information extraction ###

- Reading for comprehension.
- Reading: [Grishman](http://cs.nyu.edu/grishman/tarragona.pdf), sections 1 and 4-6
- [Slides](slides/ie-slides.pdf)

### November 13: Semi-supervised learning and domain adaptation ###

- Learning from the wrong data
- Reading: my [notes](notes/eisenstein-nlp-notes.pdf?raw=true), chapter 17. 
- Optional reading: [Jerry Zhu's survey](http://pages.cs.wisc.edu/~jerryzhu/pub/SSL_EoML.pdf);
[Jerry Zhu's book](http://www.morganclaypool.com/doi/abs/10.2200/S00196ED1V01Y200906AIM006)
- [Slides](slides/ssl-slides.pdf?raw=true)

### November 16: Final project proposals due ###

- [Independent project proposal](final-project.md) due on **November 16 at 2pm**.


### November 18: Final project check-ins ###

- [See here](final-project.md)
- Please sign up for a five-minute slot [here](https://docs.google.com/document/d/1o2nkMfxjvm3sqE8E25jUIxDQ3Zl-l4xw3_F0XoEnwhU/edit)

### November 20: Machine translation ###

- [Homework 13](homeworks/homework-13.md) due
- Reading: Collins, [IBM models 1 and 2](papers/collins-ibm12.pdf)
- Optional Reading: Chiang, [Intro to Synchronous Grammars](http://www.isi.edu/~chiang/papers/synchtut.pdf);
Lopez, [Statistical machine translation](http://www.cs.jhu.edu/~alopez/papers/survey.pdf)


### November 25: Final project lab ###

- Work in teams on final project, drop-in with Prof and TA

<!--
- Learning to process many languages at once.
- Reading: [Multisource transfer of delexicalized dependency parsers](http://www.aclweb.org/anthology-new/D/D11/D11-1006.pdf)
- Optional reading: [Cross-lingual word clusters](http://www.ryanmcd.com/papers/multiclustNAACL2012.pdf); [Climbing the tower of Babel](http://www.icml2010.org/papers/905.pdf)
-->

### November 27: Thanksgiving, no class ###

### December 2: Project presentations ###

- [See here](final-project.md)
- Initial result submissions due December 1 at 5pm.

### December 4: Current research in NLP; course wrapup ###

- [Homework 14](homeworks/homework-14.md) due
- Optional reading: [Semantic compositionality through recursive matrix-vector spaces](http://www.robotics.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf); [Vector-based models of semantic composition](http://homepages.inf.ed.ac.uk/s0453356/composition.pdf)

### Final business ###

- [See here](final-project.md)
- December 5: Initial project report due at 5PM
- December 11: Final project report due at 5PM


