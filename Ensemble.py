import pandas as pd

cat = pd.read_csv('submit/FFDCN.csv')
fmd = pd.read_csv('submit/CatBoost.csv')

cat['rating'] = cat['rating'] * 0.55 + fmd['rating'] * 0.45 # 비율은 변경할수도.
cat.to_csv('submit/Ensemble.csv', index = False)