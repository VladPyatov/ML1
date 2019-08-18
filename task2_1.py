import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier

# Download the Wine set.
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# Extract features and classes from the data.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# Create a splitter generator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Find the cross-validation classification accuracy for the k nearest neighbors method for k from 1 to 50.
gs=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=dict(n_neighbors=list(range(1, 51))), scoring='accuracy', cv=kfold)
gs.fit(X, y)
print(f"Best accuracy is {gs.best_score_:0.2f} for:")
print(gs.best_params_)

# Scale features.
X_std = scale(X)

# What value of k turned out to be optimal after reducing the features to the same scale? Did feature scaling help?
gs = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=dict(n_neighbors=list(range(1, 51))), scoring='accuracy', cv=kfold)
gs.fit(X_std, y)
print(f"Best accuracy is {gs.best_score_:0.2f} for:")
print(gs.best_params_)
