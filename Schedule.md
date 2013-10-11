Please check homeworks and projects for updates. Homeworks are not finalized until the class
before they are due. Projects are not final until the date that they are "out."

+ August 20: Welcome: History of NLP and modern applications. [Slides](lectures/lec1.pdf?raw=true)
  - Reading: Chapter 1 of
	[Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020).
	You should be able to access this PDF for free from a Georgia Tech
	computer.
+ August 22: Supervised learning: bag-of-words models and naive bayes. [Notes](lectures/lec2.pdf?raw=true)
  - [Homework 1](homeworks/homework-1.md) due
  - [Project 1](projects/proj-1/project1.md) out
  - Reading: [Chapters 0-0.3, 1-1.2 of LXMLS lab guide](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/lxmls-guide.pdf?raw=true)
  - Optional reading: [Survey on word sense disambiguation](http://promethee.philo.ulg.ac.be/engdep1/download/bacIII/ACM_Survey_2009_Navigli.pdf)
+ August 27: Discriminative classifiers: perceptron and MIRA; word-sense disambiguation. [Notes](lectures/lec3.pdf?raw=true) on perceptron; [Slides](lectures/lec3-wsd-slides.pdf?raw=true) on WSD.
  - Reading: Chapters 1.3-1.4 of [LXMLS guide](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/lxmls-guide.pdf?raw=true)
  - Reading: Parts 4-7 of [log-linear models](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-loglin.pdf?raw=true)
  - Optional reading: [Passive-aggressive learning](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
  - Optional reading: [Exponentiated gradient](http://www.cs.columbia.edu/~mcollins/papers/egjournal.pdf) training
+ August 29: Logistic regression and unsupervised learning; word sense clustering. [Notes](lectures/lec4.pdf?raw=true) on logistic regression and EM.
  - [Homework 2](homeworks/homework-2.md) due
  - Reading: [Nigam et al](http://www.kamalnigam.com/papers/emcat-mlj99.pdf)
  - Optional reading: [Expectation maximization](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-em.pdf?raw=true)
  - Optional readings: [Tutorial on EM](http://www.cc.gatech.edu/~dellaert/em-paper.pdf), [Word](http://acl.ldc.upenn.edu/P/P95/P95-1026.pdf) [sense](http://www.d.umn.edu/~tpederse/Pubs/wsdbook-2006-pedersen.pdf) [clustering](http://www.aclweb.org/anthology-new/W/W97/W97-0322.pdf)
+ September 3: More about EM; Semi-supervised learning; intro to language models[Slides](lectures/lec4-semisup-em.pdf?raw=true) on semi-supervised learning.
  [Notes](lectures/lec5.pdf?raw=true) on EM
  - [Project 1](projects/proj-1/project1.md) due
  - Reading: [Language modeling](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/collins-lm.pdf?raw=true) chapter by Michael Collins
  - Optional reading: [An empirical study of smoothing techniques for language models](http://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf), especially
	sections 2.7 and 3 on Kneser-Ney smoothing.
  - Optional reading:
    [A hierarchical Bayesian language model based on Pitman-Yor processes](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf). Requires
    some machine learning background.
  - [Word sense clustering with EM](demos/word-cluster.ipynb)
+ September 5: Language models: n-grams, smoothing, speech recognition; [Notes](lectures/lec6-lm.pdf?raw=true) on language models.
+ September 10: Finite state automata, morphology, semirings; [Notes](lectures/lec7-morphology.pdf?raw=true) on morphology; [Notes](lectures/lec7-fsa.pdf?raw=true) on FSAs
  - Reading: [Knight and May](http://ai.eecs.umich.edu/people/rounds/csli/main.pdf)
+ September 12: Finite state transduction, edit distance, finite state composition; [Notes](lectures/lec8-wfsts.pdf?raw=true) on WFSTs
  - Reading: Chapter 2 of [Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020).
  - Reading: [OpenFST slides](http://www.stringology.org/event/CIAA2007/pres/Tue2/Riley.pdf)
  - Optional reading: [Mohri and Pereira](http://dx.doi.org/10.1006/csla.2001.0184), 
  - [Homework 3](homeworks/homework-3.md) due
+ September 17: Sequence labeling 1: part-of-speech tags, hidden Markov models. [Notes](lectures/lec9-pos.pdf?raw=true); [Slides](lectures/lec9-pos-slides.pdf?raw=true)
  - Reading: [Day 2 of LXMLS](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/readings/lxmls-guide.pdf?raw=true)
  - Optional reading: [Tagging problems and hidden Markov models](http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf)
+ September 19: Sequence labeling 2: Viterbi, the Forward algorithm, and B-I-O encoding. [Notes](lectures/lec10-hmm.pdf?raw=true); [Slides on Viterbi](lectures/sequence-slides.pdf?raw=true)
  - [Homework 4](homeworks/homework-4.md) due
  - [Project 2](projects/proj-2/project2.md) out
  - Reading: [Conditional random fields](http://www.cs.columbia.edu/~mcollins/crf.pdf)
  - Optional reading: [CRF tutorial](http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)
  - Optional reading: [Discriminative training of HMMs](http://dl.acm.org/citation.cfm?id=1118694)
+ September 24: Sequence labeling 3: discriminative structure
prediction, conditional random fields, and the forward-backward
algorithm. [Slides on forward-backward](lectures/sequence-slides.pdf?raw=true); [Notes on structure perceptron](lectures/lec11-sequence-labeling.pdf?raw=true)
  - Reading: [Forward-backward](http://www.cs.columbia.edu/~mcollins/fb.pdf)
  - Optional reading: [Two decades of unsupervised POS tagging: how far have we come?](homepages.inf.ed.ac.uk/sgwater/papers/emnlp10-20yrsPOS.pdf)
+ September 26: Syntax and CFG parsing; [Notes on forward-backward](lectures/lec12-crfs.pdf?raw=true); [Notes on CFGs](lectures/lec12-cfls.pdf?raw=true)
  - Reading: [Probabilistic context-free grammars](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf)
+ October 1: Lexicalized parsing [Notes](lectures/lec13-cfg-parsing.pdf?raw=true)
  - [Project 2](project-2.md) due
  - Reading: [Lexicalized PCFGs](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/lexpcfgs.pdf)
  - Optional reading: [Accurate unlexicalized parsing](http://acl.ldc.upenn.edu/P/P03/P03-1054.pdf)
+ October 3: Dependency parsing [Notes](lectures/lec14-depparsing.pdf?raw=true); [Slides on parsing algorithms](lectures/lec14-algorithm-slides.pdf?raw=true); [Slides on PCFG failure cases](lectures/lec14-pcfg-fail.pdf?raw=true)
  - [Homework 5](homeworks/homework-5.md) due
  - Reading: [Characterizing the errors of data-driven dependency parsing models](http://acl.ldc.upenn.edu/D/D07/D07-1013.pdf)
  - Optional reading: [Eisner algorithm worksheet](http://www.cc.gatech.edu/~jeisenst/classes/cs7650_sp12/eisner_worksheet.pdf)
  - Optional reading: [Short textbook on dependency parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002), PDF should be free from a GT computer.
+ October 8: Midterm
  - [Minimal review notes](lectures/midterm-review.pdf?raw=true)
  - [Project 3](project-3.md) out
+ October 10: Midterm recap. Lexicalized parsing and alternative models of syntax.
  - [Homework 6](homeworks/homework-6.md) due
  - Reading: [The inside-outside algorithm](http://www.cs.columbia.edu/~mcollins/io.pdf)
  - Reading: [Intro to CCG](http://web.uvic.ca/~ling48x/ling484/notes/ccg_intro.pdf)
  - Optional reading: [Corpus-based induction of linguistic structure](http://acl.ldc.upenn.edu/acl2004/main/pdf/341_pdf_2-col.pdf)
  - Optional reading: [Much more about CCG](http://homepages.inf.ed.ac.uk/steedman/papers/ccg/SteedmanBaldridgeNTSyntax.pdf)
  - Optional reading: [Joshi on LTAG](http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog2805_2/pdf)
  - Optional reading: [Probabilistic disambiguation models for wide-coverage HPSG](http://acl.ldc.upenn.edu/P/P05/P05-1011.pdf)
+ October 11: Drop deadline
+ October 15: Fall recess, no class
+ October 17: Semi-supervised learning and domain adaptation.
  - Reading: [Jerry Zhu's survey](http://pages.cs.wisc.edu/~jerryzhu/pub/SSL_EoML.pdf)
  - Optional reading: [Way more about semi-supervised learning](http://www.morganclaypool.com/doi/abs/10.2200/S00196ED1V01Y200906AIM006)
+ October 22: No class. Watch [this video](http://videolectures.net/metaforum2012_pereira_semantic/)
  - [Project 3](project-3.md) due
+ October 24: Compositional semantics
  - Reading: [Manning: Intro to Formal Computational Semantics](http://www.stanford.edu/class/cs224u/readings/cl-semantics-new.pdf)
  - Optional reading: [Learning to map sentences to logical form](http://arxiv.org/pdf/1207.1420v1.pdf); 
+ October 29: Shallow semantics
  - Reading: Sections 1 and 2 of [Automatic labeling of semantic roles](http://acl.ldc.upenn.edu/J/J02/J02-3001.pdf)
  - Reading: [SRL via ILP](http://acl.ldc.upenn.edu/C/C04/C04-1197.pdf), especially section 4.
+ October 31: Distributional semantics
  - [Homework 7](homeworks/homework-7.md) due
  - [Project 4](project-4.md) out  
  - Reading: [Vector-space models](www.jair.org/media/2934/live-2934-4846-jair.pdf), sections 1, 2, 4-4.4, 6
  - Optional reading: [Semantic compositionality through recursive matrix-vector spaces](http://www.robotics.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf)
  - Optional reading: [Vector-based models of semantic composition](http://homepages.inf.ed.ac.uk/s0453356/composition.pdf)
+ November 5: Anaphora and coreference resolution
  - [Homework 8](homeworks/homework-8.md) due
  - Reading: [Multi-pass sieve](http://www.stanford.edu/~jurafsky/emnlp10.pdf)
  - Optional reading: [Large-scale multi-document coreference](http://people.cs.umass.edu/~sameer/files/largescale-acl11.pdf)
+ November 7: Discourse and dialogue
  - [Project 4](project-4.md) due
  - Reading: [Discourse structure and language technology](http://journals.cambridge.org/repo_A84ql5gR)
  - Optional: [Modeling local coherence](http://www.aclweb.org/anthology-new/J/J08/J08-1001.pdf); [Sentence-level discourse parsing](http://acl.ldc.upenn.edu/N/N03/N03-1030.pdf)
+ November 12: Project proposal presentations
  - [Project proposal](projects/indie/project.pdf?raw=true) due
  - [Homework 9](homeworks/homework-9.md) due
+ November 14: Information extraction
  - [Homework 10](homeworks/homework-10.md) due
  - Reading: [Grishman](http://cs.nyu.edu/grishman/tarragona.pdf), sections 1 and 4-6
+ November 19: Phrase-based machine translation
  - [Homework 11](homeworks/homework-11.md) due
  - Reading: [IBM models 1 and 2](papers/collins-ibm12.pdf)
  - Optional reading: [Statistical machine translation](http://www.cs.jhu.edu/~alopez/papers/survey.pdf)
+ November 21: Syntactic machine translation
  - Reading: [Intro to Synchronous Grammars](http://www.isi.edu/~chiang/papers/synchtut.pdf)
+ November 26: Multilingual learning
  - Reading: [Multisource transfer of delexicalized dependency parsers](http://www.aclweb.org/anthology-new/D/D11/D11-1006.pdf)
  - Optional reading: [Cross-lingual word clusters](http://www.ryanmcd.com/papers/multiclustNAACL2012.pdf); [Climbing the tower of Babel](http://www.icml2010.org/papers/905.pdf)
+ November 28: Thanksgiving, no class
+ December 3: Project presentations
+ December 5: Project presentations
  - [Homework 12](homeworks/homework-12.md) due at the end of class

