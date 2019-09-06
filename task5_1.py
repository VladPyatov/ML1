from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV

# Load the data from abalone.csv file.
# This is a dataset in which you want to predict the age of the shell (the number of rings) by physical measurements.
df = pd.read_csv('abalone.csv')

# Convert the sign Sex to numeric: the value of F should go to -1, I to 0, M to 1.
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Separate the contents of the files into attributes and the target variable.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train a random forest with a different number of trees: from 1 to 50 (metric - r2).
kf = KFold(5, shuffle=True, random_state=1)
clf = RandomForestRegressor(random_state=1,n_jobs=-1)
sc = make_scorer(r2_score)
gs = GridSearchCV(estimator=clf, param_grid=dict(n_estimators=list(range(1, 51))), cv=kf, n_jobs=-1, scoring='r2')
gs.fit(X, y)

# At what minimum number of trees a random forest shows quality in cross-validation above 0.52?
df = pd.DataFrame(gs.cv_results_['mean_test_score'],index=range(1, 51), columns=['r2_score'])
index = df[df['r2_score']>0.52].idxmin()
print(index.values)