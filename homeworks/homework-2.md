This homework is about word senses. Word sense disambiguation is discussed in the notes in Chapter 3.2.

First, download the [SemCor corpus](https://github.com/jacobeisenstein/gt-nlp-class/releases/tag/semcor-3.0)

Open the brown2/tagfiles directory. Select a file whose last two
digits correspond to your birthday.

The file has XML markup of text (from the Brown corpus). Each word is
marked up for its part of speech and word sense. For example,
~~~~~~
<wf cmd=done pos=VB lemma=romp wnsn=1 lexsn=2:38:00::>romping</wf>
~~~~~~
Here, “romp” is the lemma, and the word sense (wnsn) is #1.

Next, go to [WordNet](http://wordnet.princeton.edu/) and click on “Use Wordnet Online”

From this page, go to “display options” and select “show sense numbers”

Choose a single sentence from your file in the SemCor corpus. For each
word that is marked with a wnsn sense, see if there are other senses
that would be possible (according to wordnet). Say why you think a
human reader would not misinterpret the word in the text as having the
wrong sense, or say why they could make this mistake.

Finally, select a post from a blog that you like, and annotate a
single sentence for word senses as in the SemCor corpus. (Or, you may choose a Tweet instead.) You don’t need to indicate any of the attributes except POS and wnsn.

Please submit your response on t-square.
