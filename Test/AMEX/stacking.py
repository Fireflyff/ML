import glob
import pandas as pd
import numpy as np
paths = [x for x in glob.glob('/Users/yingying/Downloads/amex-default-prediction/Predictions/*.csv')]
dfs = [pd.read_csv(x) for x in paths]
dfs = [x.sort_values(by='customer_ID') for x in dfs]
for df in dfs:
    df['prediction'] = np.clip(df['prediction'], 0, 1)

# fun1
submit = pd.read_csv('/Users/yingying/Downloads/amex-default-prediction/sample_submission.csv')
submit['prediction'] = 0

for df in dfs:
    submit['prediction'] += df['prediction']

submit['prediction'] /= len(paths)

submit.to_csv('/Users/yingying/Downloads/amex-default-prediction/mean_submission_7.csv', index=None)

# fun2
submit = pd.read_csv('/Users/yingying/Downloads/amex-default-prediction/sample_submission.csv')
submit['prediction'] = 0

from scipy.stats import rankdata

for df in dfs:
    submit['prediction'] += rankdata(df['prediction']) / df.shape[0]

submit['prediction'] /= len(paths)

submit.to_csv('/Users/yingying/Downloads/amex-default-prediction/rank_submission_7.csv', index=None)