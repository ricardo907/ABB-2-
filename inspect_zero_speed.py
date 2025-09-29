import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('train.csv')
y   = df['running_speed'].astype(float).values
nom = df['nominal_speed'].astype(float).values

# 1) 名义值做 baseline
print('Baseline R2:', r2_score(y, nom))
print('Baseline MAE:', mean_absolute_error(y, nom))

# 2) 有多少明显低速的样本
ratio = y / (nom + 1e-6)
print('低速( ratio<0.7 ) 占比:', float((ratio<0.7).mean()))

# 3) 看残差分布，是否双峰
resid = y - nom
print('残差 P50/P90/P99:', pd.Series(resid).quantile([.5,.9,.99]).to_dict())
