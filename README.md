CS 4650 and 7650
==========

- **Course**: Natural Language Understanding
- **Instructor**: Jacob Eisenstein
- **Semester**: Spring 2017
- **Time**: Mondays and Wednesdays, 3:05-4:25pm
- **TAs**: Umashanthi Pavalanathan, Robert Guthrie, James Mullenbach
- [Schedule](https://docs.google.com/spreadsheets/d/178zOzls_H7LBwRRtKDhCUFd4b_pEaZVlTKI7PSj1vqw/edit?usp=sharing)
- [Synopses](https://docs.google.com/document/d/1Dr6NnaPrw6vL2g-KN28G1EjD1Kxu8BlCBVbe4CZPH7o/edit?usp=sharing) of previous classes

This course gives an overview of modern data-driven techniques for
computationally analyzing natural language. The course moves from shallow bag-of-words models to richer structural representations of how words interact to create meaning. At each level, we will discuss
the salient linguistic phemonena and most successful computational
models. Along the way we will cover machine learning techniques which
are especially relevant to natural language processing.

- [Readings](#readings)
- [Grading](#grading)
- [Help](#help)
- [Policies](#policies)

# Learning goals
<a name="learning"/>

- Acquire the fundamental linguistic concepts that are relevant to language technology. This goal will be assessed in the short homework assignments and the exams.
- Analyze and understand state-of-the-art algorithms and statistical
  techniques for reasoning about linguistic data. This goal will be
  assessed in the exams and the assigned projects.
- Implement state-of-the-art algorithms and statistical techniques for
  reasoning about linguistic data. This goal will be assessed in the
  assigned projects.
- Adapt and apply state-of-the-art language technology to new problems
  and settings. This goal will be assessed in assigned projects.
- (7650 only) Read and understand current research on natural language
  processing. This goal will be assessed in assigned projects.

The assignments, readings, and schedule are subject to change, but I will try to give as much advance notice as possible.

# Readings #
<a name="readings"/>

Readings will be drawn from my [notes](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf), from published papers and tutorials, and from the following two texts:

- [Linguistic Fundamentals for NLP](http://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020).
You should be able to access this PDF for free from a Georgia Tech computer.
- [Foundations of Statistical NLP](http://nlp.stanford.edu/fsnlp/). A PDF version is accessible through the GT library.

## Supplemental textbooks ##

These are completely optional, but might deepen your understanding of the material.

- [Speech and Language Processing](http://www.amazon.com/Speech-Language-Processing-2nd-Edition/dp/0131873210/) is the textbook most often used in NLP courses. It's a great reference for both the linguistics and algorithms we'll encounter in this course.
- [Natural Language Processing with Python](http://www.amazon.com/Natural-Language-Processing-Python-Steven/dp/0596516495)
shows how to do hands-on work with Python's Natural Language Toolkit (NLTK), and also brings a strong linguistic perspective.
- [Schaum's Outline of Probability and Statistics](http://www.amazon.com/Schaums-Outline-Probability-Statistics-Edition/dp/007179557X/ref=pd_sim_b_1?ie=UTF8&refRID=1R57HWNCW6EEWD1ZRH4C) can help you review the probability and statistics that we use in this course.

# Grading
<a name="grading"/>

The graded material for the course will consist of:

- Seven short homework assignments, of which you must do five. Most of these involve performing linguistic annotation on some text of your choice. The purpose is to get a basic understanding of key linguistic concepts. Each assignment should take less than an hour. Each homework is worth 2 points (10 total). 
- Five assigned problem sets. These involve building and using NLP techniques which are at or near the state-of-the-art. The purpose is to learn how to implement natural language processing software, and to have fun. These assignments must be done individually. Each problem set is worth ten points (50 total). Students enrolled in CS 7650 will have an additional, research-oriented component to the problem sets.
- An in-class midterm exam, worth 20 points, and a final exam, worth 20 points. The purpose of these exams are to assess understanding of the core theoretical concepts, and to encourage you to review and synthesize your understanding of these concepts. Barring a personal emergency or an institute-approved absence, you must take the exam on the days indicated in the schedule. See [here](http://www.deanofstudents.gatech.edu/content/25/absences) and [here](http://www.registrar.gatech.edu/students/formlanding/iaabsences.php) for more information on GT policy about absences.

## Late policy

Problem sets will be accepted up to 72 hours late, at a penalty of 20% per 24 hours. (Maximum score after missing the deadline: 8/10; maximum score 24 hours after the deadline: 6/10, etc.)  It is usually best just to turn in what you have at the due date. Late homeworks will not be accepted. This late policy is intended to ensure fair and timely evaluation.

# Getting help
<a name="help"/>

## Office hours

- James: Mon 10:30-11:30 (CCB commons)
- Jacob: Tues 3-4pm (CCB 316)
- Umashanthi: Wed 1:00 - 2:00 (CCB commons)
- Robert: Thurs 9:30-10:30 (CCB commons)

Robert will meet in the COC common area. We're hoping to get CCB 360B for Umashanthi and James' office hours, but it's not finalized yet.

## Online help

Please use Piazza rather than personal email to ask questions. This helps other students, who may have the same question. Personal emails may not be answered. If you cannot make it to office hours, please use Piazza to make an appointment. It is unlikely that I will be able to chat if you make an unscheduled visit to my office. The same is true for the TAs.

# Class policies
<a name="policies"/>

Attendance will not be taken, but **you are responsible for knowing what happens in every class**. I will try to post slides and notes online, and to share announcements on T-Square, but there are no guarantees. So if you cannot attend class, make sure you check up with someone who was there.

Respect your classmates and your instructor by avoiding distractions. This means be there on time, turn off your cellphone, and save side conversations for after class. Also, if you can't read something I wrote on the board, or if you think I made a mistake in a derivation, please raise your hand and tell me!

[Multiple studies](http://www.newyorker.com/online/blogs/elements/2014/06/the-case-for-banning-laptops-in-the-classroom.html) have shown that using a laptop in class -- even for taking notes -- reduces students' educational attainment. I suggest you try pen and paper for a few weeks, and see if it helps you concentrate. Whatever technology you decide to use, it is your responsibility to ensure that it does not distract your classmates or me.

## Prerequisites
<a name="prerequisites"/>

The official prerequisite for CS 4650 is CS 3510/3511, "Design and Analysis of Algorithms." This prerequisite is essential because understanding natural language processing algorithms requires familiarity with dynamic programming, as well as automata and formal language theory: finite-state and context-free languages, NP-completeness, etc. While course prerequisites are not enforced for graduate students, prior exposure to analysis of algorithms is very strongly recommended.

Furthermore, this course assumes:

- Good coding ability, corresponding to at least a third or
  fourth-year undergraduate CS major. Assignments will be in Python.
- Background in basic probability, linear algebra, and calculus.
- Familiarity with machine learning is *helpful but not assumed*. Of
  particular relevance are linear classifiers: perceptron, naive
  Bayes, and logistic regression.

People sometimes want to take the course without having all of these
prerequisites. Frequent cases are:

- Junior CS students with strong programming skills but limited
  theoretical and mathematical background,
- Non-CS students with strong mathematical background but limited
  programming experience.

Students in the first group suffer in the exam and don't understand
the lectures, and students in the second group suffer in the problem sets. My advice is to get the background material first, and
then take this course.

## Collaboration policy

One of the goals of the assigned work is to assess your individual progress in meeting the learning objectives of the course. You may discuss the homework and projects with other students, but your work must be your own -- particularly all coding and writing. For example:

### Examples of acceptable collaboration

- Alice and Bob discuss alternatives for storing large, sparse vectors of feature counts, as required by a problem set.
- Bob is confused about how to implement the Viterbi algorithm, and asks Alice for a conceptual description of her strategy.
- Alice asks Bob if he encountered a failure condition at a "sanity check" in a coding assignment, and Bob explains at a conceptual level how he overcame that failure condition.
- Alice is having trouble getting adequate performance from her part-of-speech tagger. She finds a blog page or research paper that gives her some new ideas, which she implements.

### Examples of unacceptable collaboration

- Alice and Bob work together to write code for storing feature counts.
- Alice and Bob divide the assignment into parts, and each write the code for their part, and then share their solutions with each other to complete the assignment.
- Alice or Bob obtain a solution to a previous year's assignment or to a related assignment in another class, and use it as the starting point for their own solutions.
- Bob is having trouble getting adequate performance from his part-of-speech tagger. He finds source code online, and copies it into his own submission.
- Alice wants to win the Kaggle competition for a problem set. She finds the test set online, and customizes her submission to do well on it.

Some assignments will involve written responses. Using other people’s text or figures without attribution is plagiarism, and is never acceptable.

Suspected cases of academic misconduct will be (and have been!) referred to the Honor Advisory Council. For any questions involving these or any other Academic Honor Code issues, please consult me, my teaching assistants, or http://www.honor.gatech.edu.
