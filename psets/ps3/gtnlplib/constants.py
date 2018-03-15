# Data files
EN_TRAIN_FILE = "data/en-ud-train.txt"
EN_DEV_FILE = "data/en-ud-dev.txt"
EN_TEST_FILE = "data/en-ud-test-hidden.txt"
NR_TRAIN_FILE = "data/no_bokmaal-ud-train.txt"
NR_DEV_FILE = "data/no_bokmaal-ud-dev.txt"
NR_TEST_FILE = "data/no_bokmaal-ud-test-hidden.txt"
PRETRAINED_EMBEDS_FILE = "data/pretrained-embeds.pkl"

# Prediction output files
EN_D3_2_DEV_FILENAME = "en_3_2-dev.preds"
EN_D3_2_TEST_FILENAME = "en_3_2-test.preds"
NR_D3_3_DEV_FILENAME = "no_3_3-dev.preds"
NR_D3_3_TEST_FILENAME = "no_3_3-test.preds"

EN_D4_6_DEV_FILENAME = "en_4_6-dev.preds"
EN_D4_6_TEST_FILENAME = "en_4_6-test.preds"
NR_D4_7_DEV_FILENAME = "no_4_7-dev.preds"
NR_D4_7_TEST_FILENAME = "no_4_7-test.preds"

EN_BAKEOFF_FILENAME = "bakeoff-dev-en.preds"
NR_BAKEOFF_FILENAME = "bakeoff-dev-nr.preds"

# Keys
EN_DEV_GOLD = "data/en-ud-dev-key.txt"
NR_DEV_GOLD = "data/no_bokmaal-ud-dev-key.txt"

class Actions:
    """Simple Enum for each possible parser action"""
    SHIFT = 0
    ARC_L = 1
    ARC_R = 2

    NUM_ACTIONS = 3

    action_to_ix = { "SHIFT": SHIFT,
                     "ARC_L": ARC_L,
                     "ARC_R": ARC_R }
    ix_to_action = {i:a for a,i in action_to_ix.items()}

END_OF_INPUT_TOK = "<END-OF-INPUT>"
NULL_STACK_TOK = "<NULL-STACK>"
ROOT_TOK = "<ROOT>"

HAVE_CUDA = False
