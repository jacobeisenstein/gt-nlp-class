# 3.1 (0.5 points)

Fill in the rest of the table below:

|      | they | can | can | fish | END |
|------|------|-----|-----|------|-----|
| Noun | -2   | -10 | -10 | -15  | n/a |
| Verb | -13  | -6  | -11 | -16  | n/a |
| End  | n/a  | n/a | n/a | n/a  | -17 |


# 4.3 (0.5 points)

Do you think the predicted tags "PRON AUX AUX NOUN" for the sentence "They can can fish" are correct? Use your understanding of parts-of-speech from the notes.
No, it's wrong. The thrid word is a verb. Because aux doesn't go after another aux in english grammar and "can" can possibly be a verb. And in this case,I'm guessing the reason that two "can"s are all labeled "AUX" is the emission weight or prior probability for "can" being labeled as "AUX" is really large.

# 4.4 (0.5 points)

The HMM weights include a weight of zero for the emission of unseen words. Please explain:

- why this is a violation of the HMM probability model explained in the notes;
If the emission model for unseen words are 0, then log(0) = 1.0, and we would get a sum of emission weight larger than 1, which violates the rule that sum of weights should be 1.0.

- How, if at all, this will affect the overall tagging.
In this case, the emission weight won't affect the labeling because the emission part is 0, and only transition weight will count in scoring.

# 5.1 (1 point 4650; 0.5 points 7650)

Please list the top three tags that follow verbs and nouns in English and Japanese.
English [after verb]: DET, ADP, PRON
Japanese [after verb]: NOUN, --END--, PUNCT

English [after noun]: PUNCT, ADP, NOUN
Japanese [after noun]: NOUN, VERB, PUNCT
Try to explain some of the differences that you observe, making at least two distinct points about differences between Japanese and English.

1. In english, people usually put somehing such as a DET or a NOUN after verb, for example "I saw that car.", which put DET after verb. And some verbs are intransitive so we put ADP after verb, for instance: "look at". 
However, in Japanese, people usually put verb at the end of a sentence.

2. In english, the noun can be at the end of a sentence or followed by some description, for example "I eat fish." and "I eat the fish that was bought yesterday." 
But in Japanese, the noun usually goes before verb or noun or put at the end of the sentence, for example "I fish eat." if a japanese sentence is directly translated from japanese.

# 6 (7650 only; 1 point)

Find an example of sequence labeling for a task other than part-of-speech tagging, in a paper at ACL, NAACL, EMNLP, EACL, or TACL, within the last five years (2012-2017). 

## List the title, author(s), and venue of the paper.

## What is the task they are trying to solve?

## What tagging methods do they use? HMM, CRF, max-margin markov network, something else?

## Which features do they use?

## What methods and features are most effective?
