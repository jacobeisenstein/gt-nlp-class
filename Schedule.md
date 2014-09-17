CS 4650 and CS 7650 will meet jointly, on Tuesdays and Thursdays from 3:05 - 4:25PM, in College of Computing 101.

**This is a (permanently) provisional schedule.** Readings, notes, slides, and homework will change. Readings and homeworks are final at the time of the class **before** they are due (e.g., thursdays readings are final on the preceding tuesday); problem sets are final on the day they are "out." Please check for updates until then.


### August 19: Welcome ###

- History of NLP and modern applications. Review of probability. 
- **Reading**: Chapter 1 of [Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020).
You should be able to access this PDF for free from a Georgia Tech computer.
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
- [Slides](slides/lec9-hmm-slides.pdf?raw=true)

### September 18: Sequence labeling 2 ###

- Viterbi, the Forward algorithm, and B-I-O encoding. 
- [Homework 5](homeworks/homework-5.md) due
- Reading: [Conditional random fields](http://www.cs.columbia.edu/~mcollins/crf.pdf)
- Optional reading: [CRF tutorial](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf); [Discriminative training of HMMs](http://dl.acm.org/citation.cfm?id=1118694)

### September 23: Sequence labeling 3 ###

- Discriminative structure prediction, conditional random fields, and the forward-backward algorithm.
- [Problem set 2a](psets/ps2/ps2.md) due
- [Problem set 2b](psets/ps2/ps2.md) out
- Reading: [Forward-backward](http://www.cs.columbia.edu/~mcollins/fb.pdf)
- Optional reading: [Two decades of unsupervised POS tagging: how far have we come?](homepages.inf.ed.ac.uk/sgwater/papers/emnlp10-20yrsPOS.pdf)

### September 25: Syntax and CFG parsing ###

- Context-free grammars; constituency; parsing
- [Homework 6](homeworks/homework-6.md) due
- Reading: [Probabilistic context-free grammars](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf) and possibly my notes.

### September 30: Dependency parsing ###

- [Problem set 2b](psets/ps2/ps2.md) due
- Reading: my notes
- Optional reading: [Eisner algorithm worksheet](http://www.cc.gatech.edu/~jeisenst/classes/cs7650_sp12/eisner_worksheet.pdf);
[Characterizing the errors of data-driven dependency parsing models](http://acl.ldc.upenn.edu/D/D07/D07-1013.pdf);
[Short textbook on dependency parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002), PDF should be free from a GT computer.


### October 2: Catch-up, midterm review ###

- [Homework 7](homeworks/homework-7.md) due

### October 7: Midterm ###

- [Minimal review notes](lectures/midterm-review.pdf?raw=true) from 2013

### October 9: Midterm recap and lexicalized parsing ###
- [Notes](lectures/lec13-cfg-parsing.pdf?raw=true)
- [Problem set 3](psets/ps3/ps3.md) out
- Reading: [Lexicalized PCFGs](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/lexpcfgs.pdf)
- Optional reading: [Accurate unlexicalized parsing](http://acl.ldc.upenn.edu/P/P03/P03-1054.pdf)

### October 10: Drop deadline ###

### October 14: Fall recess, no class ###

### October 16: Modern competitive parsing ###

- [Homework 8](homeworks/homework-8.md) due

### October 21: Alternative models of syntax ###

- Mostly CCG, but a little about L-TAG and and HPSG.
- [Problem set 3](psets/ps3/ps3.md) due
- Reading: likely my notes, unless I can find something good
- Optional reading: [Intro to CCG](http://web.uvic.ca/~ling48x/ling484/notes/ccg_intro.pdf);
[The inside-outside algorithm](http://www.cs.columbia.edu/~mcollins/io.pdf); 
[Corpus-based induction of linguistic structure](http://acl.ldc.upenn.edu/acl2004/main/pdf/341_pdf_2-col.pdf);
[Much more about CCG](http://homepages.inf.ed.ac.uk/steedman/papers/ccg/SteedmanBaldridgeNTSyntax.pdf); [LTAG](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog2805_2/pdf); [Probabilistic disambiguation models for wide-coverage HPSG](http://acl.ldc.upenn.edu/P/P05/P05-1011.pdf)

### October 23: Compositional logical semantics ###

- [Homework 9](homeworks/homework-9.md) due
- Reading: [Manning: Intro to Formal Computational Semantics](http://www.stanford.edu/class/cs224u/readings/cl-semantics-new.pdf)
- Optional reading: [Learning to map sentences to logical form](http://arxiv.org/pdf/1207.1420v1.pdf); 

### October 28: Shallow semantics ###

- Frame semantics, and semantic role labeling. 
- [Homework 10](homeworks/homework-10.md) due
- Reading: likely my notes.
- Optional reading:  [Automatic labeling of semantic roles](http://acl.ldc.upenn.edu/J/J02/J02-3001.pdf);
[SRL via ILP](http://acl.ldc.upenn.edu/C/C04/C04-1197.pdf).
- Optional [video](http://videolectures.net/metaforum2012_pereira_semantic/)

### October 30: Distributional semantics ###

- Vector semantics, latent semantic indexing, neural word embeddings
- [Problem set 4](psets/ps4/ps4.md) out
- Reading: [Vector-space models](www.jair.org/media/2934/live-2934-4846-jair.pdf), sections 1, 2, 4-4.4, 6

### November 4: Anaphora and coreference resolution ###

- Knowing who's on first. [Notes](lectures/lec20-coref-notes.pdf?raw=true) [Slides](lectures/lec20-coref-slides.pdf?raw=true)
- [Homework 11](homeworks/homework-11.md) due
- Reading: likely my notes
- Option reading: [Multi-pass sieve](http://www.stanford.edu/~jurafsky/emnlp10.pdf);
[Large-scale multi-document coreference](http://people.cs.umass.edu/~sameer/files/largescale-acl11.pdf)

### November 6: Discourse and dialogue ###

- Coherence; discourse connectives; rhetorical structure theory; speech acts. 
- [Homework 12](homeworks/homework-12.md) due
- [Problem set 4](psets/ps4/ps4.md) due
- Reading: likely my notes 
- Optional: [Discourse structure and language technology](http://journals.cambridge.org/repo_A84ql5gR);
[Modeling local coherence](http://www.aclweb.org/anthology-new/J/J08/J08-1001.pdf); [Sentence-level discourse parsing](http://acl.ldc.upenn.edu/N/N03/N03-1030.pdf)

### November 11: Information extraction ###

- Reading for comprehension.
[Notes](lectures/lec22-ie-notes.pdf?raw=true) [Slides](lectures/lec22-ie.pdf?raw=true)
- Reading: [Grishman](http://cs.nyu.edu/grishman/tarragona.pdf), sections 1 and 4-6

### November 13: Semi-supervised learning and domain adaptation ###

- Learning from the wrong data
- Reading: likely my notes 
- [Independent project proposal](indie.md) due
- Optional reading:
- [Jerry Zhu's survey](http://pages.cs.wisc.edu/~jerryzhu/pub/SSL_EoML.pdf);
[Jerry Zhu's book](http://www.morganclaypool.com/doi/abs/10.2200/S00196ED1V01Y200906AIM006)
  
### November 18: Phrase-based machine translation ###

- [Homework 13](homeworks/homework-13.md) due
- Reading: [IBM models 1 and 2](papers/collins-ibm12.pdf)
- Optional reading: [Statistical machine translation](http://www.cs.jhu.edu/~alopez/papers/survey.pdf)

### November 20: Syntactic machine translation ###

- Reading: [Intro to Synchronous Grammars](http://www.isi.edu/~chiang/papers/synchtut.pdf)

### November 25: Multilingual NLP (or maybe deep learning) ###

- Learning to process many languages at once.
- Reading: [Multisource transfer of delexicalized dependency parsers](http://www.aclweb.org/anthology-new/D/D11/D11-1006.pdf)
- Optional reading: [Cross-lingual word clusters](http://www.ryanmcd.com/papers/multiclustNAACL2012.pdf); [Climbing the tower of Babel](http://www.icml2010.org/papers/905.pdf)

### November 27: Thanksgiving, no class ###

### December 2: Project presentations ###

- Initial result submissions due December 1 at 5pm.

### December 4: Current research in NLP; course wrapup ###

- [Homework 14](homeworks/homework-14.md) due
- Optional reading: [Semantic compositionality through recursive matrix-vector spaces](http://www.robotics.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf); [Vector-based models of semantic composition](http://homepages.inf.ed.ac.uk/s0453356/composition.pdf)

### Final business ###

- December 5: Initial project report due at 5PM
- December 11: Final project report due at 5PM


