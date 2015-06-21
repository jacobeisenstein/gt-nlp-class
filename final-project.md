Win the final project
=================

In your final project, you will reimplement a strong baseline model for Rhetorical Structure Theory discourse parsing, and then try to add some improvements or new applications. You will build on Yangfeng Ji's [implementation](https://github.com/jiyfeng/RSTParser) of shift-reduce parsing for RST. (Note: we may continue to make small changes to this implementation. If so, we'll let you know.)

This is a team project, with up to three people per team. Expectations are higher for larger teams. This project is worth 14\% of your final grade.

# On RST #

You can start by reading up on rhetorical structure theory. It is described in several readings:

- [Discourse structure and language technology](http://journals.cambridge.org/repo_A84ql5gR), which was the assigned reading for November 4
- The [RST Website](http://www.sfu.ca/rst/index.html), which contains some example analyses and other good information
- The classic paper from [Mann and Thompson](http://www.sfu.ca/rst/05bibliographies/bibs/Mann_Thompson_1988.pdf)
- This [survey](https://www.sfu.ca/~mtaboada/docs/Taboada_Mann_RST_Part1.pdf) from Taboada and Mann

These readings may give you some ideas for the independent part of the project.

# Your timeline #

- **Right now**: start thinking about the independent part of your project
- **Sunday, November 16**: submit 1-2 page proposal at 2pm. Explain what you want to do in the independent part of the project. (2 points)
- **Tuesday, November 18**: schedule a 5-minute meeting with me during class time or office hours to talk about your project. [Sign up here](https://docs.google.com/document/d/1o2nkMfxjvm3sqE8E25jUIxDQ3Zl-l4xw3_F0XoEnwhU/edit?usp=sharing).
- **Tuesday, December 2**: present results in class. (2 points)
- **Friday, December 5**: project report due at 5pm. (10 points)
- **Thursday, December 11**: optionally, submit updated version of project report. 

Late assignments will be marked in the same way as problem sets: 20% deduction per 24 hours, with a maximum of 72 hours late before the assignment is no longer accepted.

# Compulsory part of the project #

Replicate all features from Table 2 of [Ji and Eisenstein, 2014](http://www.cc.gatech.edu/~jeisenst/papers/ji-acl-2014.pdf). You can skip the last feature, because the RST implementation doesn't make it easy to see which sentence each span is in. As Yangfeng showed, nearly all of your work will be in the feature.py file.

However, computing these features will require knowing the part-of-speech tags and dependency parses for the RST treebank data. You should use an existing library for this. Here are some suggestions:

- Stanford's [CoreNLP](http://nlp.stanford.edu/software/corenlp.shtml) package is very mature and well-supported. You can run this on the command-line and save the output for your data, or you can try to get the python bindings to work (e.g., [these](https://github.com/dasmith/stanford-corenlp-python) or [these](https://pypi.python.org/pypi/corenlp-python)).
- [MaltParser](http://www.maltparser.org/) is a good, fast dependency parser. Apparently this can now be called from [nltk](http://www.nltk.org/_modules/nltk/parse/malt.html), although there isn't much documentation that I can find.
- Here is a [pure python implementation](http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing/) of dependency parsing, and here is the implementation for [POS tagging](http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/). The only potential problem here is that you might not be able to get the trained models, so you'd have to train them yourselves (and you don't have the data). But the models might be out there.

As with previous bakeoffs, you will submit your predicted parses on the test data, and we will evaluate them. You will also submit your code.

Please use an iPython notebook to show how you have implemented this part of the project. Compute your parser accuracy with all features, and also perform ablation tests in which each feature is removed. You can compute the POS tags and dependency parses off-line.

[Here](psets/final_project/tips-for-compulsory-part.md) are some pointers about implementing the compulsory part of the project. 

# Freestyle part of the project #

This is your chance to be creative and try something new. Start early and make a good plan: time spent thinking through your project is worth more than time spent hacking on a bad idea later.

Almost any area of the class that we have covered so far could be relevant to discourse parsing. Here are some possible directions.

## Improve RST parsing through better features ##

One reason that relation detection in RST parsing is particularly difficult is because it is a fundamentally *semantic* problem. Yet the RST Treebank is fairly small, so simplistic data-hungry approaches like bilexical features are likely to overfit.

If you plan to add features, read up on prior work before you start. It's okay to reimplement a previous paper, but you want to be aware of what you're doing. Relevant readings are listed below. Note that the parsing and learning models may be different, but many of the features are still applicable. Note also that to get this to work, you may have to modify the code more deeply than just in feature.py.

There are many types of features that you could consider:

- **Word representations**: In problem set 4, you induced various types of word representations. It's also possible to download discriminatively-trained word representations, which may work better ([here](http://www-nlp.stanford.edu/projects/glove) and [here](http://code.google.com/p/word2vec/). We tried something like this in [our paper](http://www.cc.gatech.edu/grads/y/yji37/papers/ji-acl-2014.pdf), too. Just plugging in word vectors isn't very interesting or difficult, so I would expect something more: a careful comparison of different feature types, an analysis of the types of cases where this helps and doesn't help, etc.
- **Lexical semantics**: A related point is that WordNet and VerbNet could be useful resources, generalizing word-level features into synsets or frames. *Word morphology* might also be relevant, since it might convey temporal information, hypotheticals, etc.
- **Shallow semantics**: A more structured semantic representation, such as Framenet, could make a big difference in discourse parsing. You can find a FrameNet parser [here](https://code.google.com/p/semafor-semantic-parser/wiki/FrameNet), but you'd have to think hard about how it could be useful. Semantic Role Labeling could also be useful: you could build features that relate to predicates and arguments, rather than just words. Here is a downloadable [SRL system](http://ml.nec-labs.com/senna/); there are others online as well.
- **Entity chains and lexical chains**: If two discourse units discuss the same entity or action, this is a good clue that they are related. You could download a coreference system and build features based on entity chains. Lexical chains could also be relevant. Of particular interest are words and entities that are rare in the discourse, not ones which occur in every sentence. This also relates to the "entity grid" idea from centering theory, which we discussed in class.
- **Topic segmentation**: Running a topic segmentation algorithm like [Bayes-seg](https://github.com/jacobeisenstein/bayes-seg) could provide features that are useful for the structural phase of RST parsing, since you would be less likely to merge spans across segments.
- **Topic modeling and LSA**: Topic models and latent semantic analysis could give vector descriptions of the topics discussed by each span; presumably you'd prefer to merge spans that are on the same topics.
- **Paraphrase**: The Penn Paraphrase Database [PPDB](http://www.cis.upenn.edu/~ccb/ppdb/) contains zillions of related phrases. Spans that contain near paraphrases should likely be merged first.
- **Feature combination**: The Sagae paper went crazy with lots of feature combinations, but didn't sufficiently explore how much of a difference they made. A good exploration of this issue, along with a few additional features above the baseline, could be a good project.

There are surely other features too. In any case, in your proposal I want to know what type of feature you plan to design, what types of phenomena you expect it to help address (with examples, if possible), and how you will measure whether it worked.

## Improve RST parsing through better learning or algorithms ##

Alternatively, you could stick with the baseline features, and try to improve the learning or the parsing algorithms.

- Our code uses the default implementation of the support vector machine classifier from [scikit learn](http://scikit-learn.org/). It is very likely that other learning algorithms could do better. (However, naively comparing a bunch of learning algorithms without thinking about why you expect one to be better is not a very interesting project.)
- Better regularization or feature selection could also help. The Feng & Hirst paper (below) found feature selection to be very useful.
- Our code does simple shift-reduce parsing, but it is likely that beam search could do better. A more advanced search algorithm like A* would be even more interesting.
- If you can implement beam search, you could try to find the best parse by **reranking** the top-K outputs. We saw reranking in the "competitive parsing" lecture; as far as I know, it has never been applied to RST parsing.

Please be aware, many of these ideas may require substantially changing the parser code. You are welcome to make any changes you like.

## Cool applications of RST Parsing ##

As discussed in class, RST parsing could be applied to many document-level analysis tasks, such as sentiment analysis, summarization, topic classification, political ideology detection, recommendation, etc. Applying your baseline model (from the compulsory part) to a new dataset with a specific task in mind could be a great project.

However, **you have to have a quantitative way to measure whether it worked.** In practice, this means that you need a substantial dataset that is already labeled for your judgment of interest (sentiment, relevance, summary, whatever). Do not propose to label your own data for this part of the project.

## A note about results ##

Not every idea is going to work, and class projects are a "safe space" to try new things. But it's important that your proposal contains a convincing intuition for why your idea *should* work, and that your writeup contains some evidence that you executed your idea correctly. Here are three scenarios:

- 1. You make a convincing proposal, I tell you it seems reasonable.  You get poor results, but you convince me you executed your idea correctly, and tried reasonable fall-back alternatives.
- 2. You make a convincing proposal, but do not convince me that you executed your idea correctly.
- 3. You make an unconvincing proposal, and I suggest you try something else. You persist in your original idea -- or move to another unconvincing idea -- and get poor results.

Scenario 1 will be graded most favorably, and scenario 3 least favorably.

Similarly, getting good results does not ensure a good grade. Research is not sports. You need to have some explanation for why your system works.

# Presentation and writeup #

On December 2, you will briefly present your results in class. **This means you need to have results by December 2.** A presentation that says "here's what we hope to do" will get poor marks. Based on your presentation, your classmates and I will offer feedback make your project even better. The presentation is worth 2 points, and should focus on the independent part of the project. **Submit your slides on T-Square in PDF format by 1pm on December 2. No more than two slides per team.**

The writeup for the compulsory part should include an iPython notebook showing the dev set scores for your features. For the freestyle part, your writeup should explain your high-level idea, why you thought it would work, how you executed your plan, and how you evaluated it. You can do this with an iPython notebook, or with a more conventional writeup. In either case, you need to use text and figures to make your point.

Of the 10 points for the writeup, the compulsory part will be worth 7 points for individual submissions, 5 points for two-person teams, and 3 points for three-person teams. The freestyle part will be worth 3 points for individual submissions, 5 points for two-person teams, and 7 points for three-person teams. This reflects the fact that I expect larger teams to have to spend less time per person on the compulsory part, and therefore to have more time for the independent part. However, note that all teams get two points for the presentation and two points for the proposal, regardless of size.

On December 5, you will submit your initial project writeup. I will try to grade it quickly. You can resubmit on December 11; if so, your writeup grade will be the average of the two submissions.

# Papers about RST #

Please check out these papers for more ideas about RST parsing.

- [Sentence-level discourse parsing ...](http://www.radusoricut.com/pubs/soricut-marcu-naacl2003.pdf)
- [A decision-based approach to rhetorical parsing](http://www.aclweb.org/anthology/P99-1047)
- [A novel discourse parser ...](http://dl.acm.org/citation.cfm?id=1690239)
- [Analysis of discourse structure with syntactic ...](http://dl.acm.org/citation.cfm?id=1697253)
- [Veins theory ...](http://hal.inria.fr/docs/00/52/18/89/PDF/newColing8.pdf)
- [An effective Discourse Parser that uses Rich Linguistic Information](http://csgsa.cs.uic.edu/PS-papers/N09-1064.pdf)
- [Text-level discourse parsing with rich linguistic features](ftp://ftp.white.toronto.edu/dist/gh/Feng+Hirst-ACL-2012.pdf)
- [A novel discriminative framework...](http://www.aclweb.org/anthology-new/D/D12/D12-1083.pdf)
- [Recursive deep models for discourse parsing](http://www.aclweb.org/anthology/D14-1220)
- [Representation learning for text-level discourse parsing](http://www.cc.gatech.edu/grads/y/yji37/papers/ji-acl-2014.pdf)
