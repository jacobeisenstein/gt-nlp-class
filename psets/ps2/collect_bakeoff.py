import codecs
import pandas as pd

preds=[]
filename='model-te-nr.preds'
with codecs.open(filename,encoding='utf8') as f:
	for i,line in enumerate(f):
		content = line.strip()
		if len(content)>0:
			preds.append(content)

predLabel = pd.DataFrame(preds)
predLabel.index += 1
predLabel.index.name = 'id'
predLabel.columns = ['tag']

predLabel.to_csv("bakeoff_submission.csv")