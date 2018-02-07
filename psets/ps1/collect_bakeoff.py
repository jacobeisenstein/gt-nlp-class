import numpy as np
import pandas as pd

label_set = np.asarray(['1980s', '1990s', '2000s', 'pre-1980'])
preds = np.load('bakeoff-test.preds.npy')

predLabel = pd.DataFrame(label_set[preds])
predLabel.index += 1
predLabel.index.name = 'ID'
predLabel.columns = ['Era']

predLabel.to_csv("bakeoff_submission.csv")
