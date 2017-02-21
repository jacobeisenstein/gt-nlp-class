# You may store the dataset anywhere you want.
# Make the appropriate changes to the following constants.

# please leave the correct grading scheme uncommented
GRADING='4650'
#GRADING='7650'

TRAIN_FILE = 'data/en-ud-simpler-train.conllu'
DEV_FILE = 'data/en-ud-simpler-dev.conllu'

TEST_FILE_HIDDEN = 'data/en-ud-simpler-test-hidden.conllu'
TEST_FILE = 'data/en-ud-simpler-test.conllu' # You do not have this

JA_TRAIN_FILE = 'data/ja-ud-simpler-train.conllu'
JA_DEV_FILE = 'data/ja-ud-simpler-dev.conllu'
JA_TEST_FILE = 'data/ja-ud-simpler-test.conllu' # You do not have this
JA_TEST_FILE_HIDDEN = 'data/ja-ud-simpler-test-hidden.conllu'

OFFSET = '**OFFSET**'
UNKNOWN  = '**UNKNOWN**'

START_TAG = '--START--'
TRANS ='--TRANS--'
END_TAG = '--END--'
EMIT = '--EMISSION--'

PRE_START_TOKEN = '[[START]]'
POST_END_TOKEN = '[[END]]'

PREV_WORD_FEAT='--PREV-WORD--'
PREV_TAG_FEAT='--PREV-TAG--'
CURR_WORD_FEAT='--CURR-WORD--'
NEXT_WORD_FEAT='--NEXT-WORD--'
SUFFIX_FEAT='--SUFFIX--'
