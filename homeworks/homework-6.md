Syntactic ambiguity and parsing
----------

In this homework, you will analyze some syntactic structures in the
Twitter part-of-speech corpus. You will select sentences from the
training file oct27.train. You can select any sentences you like, but
you may find it easiest to work with relatively short sentences. You
may consult the Stanford parser to get some sense of how an automatic
parser might treat a sentence like yours, but keep in mind that like
all automatic parsers, it makes mistakes.

# Prepositional phrase attachment #

Our canonical example “They eat fish with chopsticks” has
prepositional phrase attachment ambiguity. Find two examples of
prepositional phrase attachment ambiguity in the corpus, choose the
attachment site that you feel is correct, and explain why. You can
find prepositions by searching for the “P” tag. However, unlike the
Penn Treebank, this corpus tags “to” as “P”, even when it is
functioning in an infinitive verb phrase (e.g., “we are going to ace
the midterm”). You are more likely to find prepositional phrases by
searching for other prepositions, like “in”, “at”, “of”, etc.

# Coordination scope #

Another common source of syntactic ambiguity is the scope of
coordinators like “and” and “or”. For example, in the sentence “She
likes English music and tacos”, there are two interpretations:

- She likes ((English music) and (tacos))
- She likes (English (music and tacos))

Find two examples of scope ambiguity in the Twitter dataset, showing the possible interpretations and indicating which one you think is correct.

# Full parsing #

You have now gathered four examples. Provide full syntactic analysis
for two of them, using this [grammar fragment](http://www.cse.buffalo.edu/~rapaport/675w/cfg.pdf). You may need to make up a few
productions of your own to deal with Twitter-specific phenomena like
hashtags.
