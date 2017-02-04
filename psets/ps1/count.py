from gtnlplib import preproc

y_tr,x_tr = preproc.read_data('reddit-dev.csv', #filename
                                       'subreddit', #label field
                                       preprocessor=preproc.tokenize_and_downcase) #your preprocessor

corpus_counts = preproc.get_corpus_counts(x_tr)
print corpus_counts.most_common(30)
