import pandas as pd
import csv

clickbait_train = pd.read_csv('train/clickbait.tsv', delimiter='\t')
non_clickbait_train = pd.read_csv('train/non_clickbait.tsv', delimiter='\t')
clickbait_dev = pd.read_csv('dev/clickbait.tsv', delimiter='\t')
non_clickbait_dev = pd.read_csv('dev/non_clickbait.tsv', delimiter='\t')
clickbait_test = pd.read_csv('test/clickbait.tsv', delimiter='\t')
non_clickbait_test = pd.read_csv('test/non_clickbait.tsv', delimiter='\t')

clickbait_train['label'] = 1
non_clickbait_train['label'] = 0
clickbait_dev['label'] = 1
non_clickbait_dev['label'] = 0
clickbait_test['label'] = 1
non_clickbait_test['label'] = 0

train = pd.concat([clickbait_train, non_clickbait_train], axis=0)
dev = pd.concat([clickbait_dev, non_clickbait_dev], axis=0)
test = pd.concat([clickbait_test, non_clickbait_test], axis=0)

train = train[['id', 'origin', 'content', 'label']]
dev = dev[['id', 'origin', 'content', 'label']]
test = test[['id', 'origin', 'content', 'label']]

train = train.sample(frac=1).reset_index(drop=True)
dev = dev.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

train.to_csv('train/train.tsv', sep='\t', index=False, escapechar="|", quoting=csv.QUOTE_NONE)
dev.to_csv('dev/dev.tsv', sep='\t', index=False, escapechar="|", quoting=csv.QUOTE_NONE)
test.to_csv('test/test.tsv', sep='\t', index=False, escapechar="|", quoting=csv.QUOTE_NONE)
