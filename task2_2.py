import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston

# Download the Boston set.
ds_boston = load_boston()

# Scale features.
ds_boston.data = scale(ds_boston.data)

# Create a splitter generator.
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Go through the different options for the metric parameter p over the grid
# from 1 to 10 in such a way that 200 options are tested in total
gs2 = GridSearchCV(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance'), param_grid=dict(p=np.linspace(1,10,200)), scoring='neg_mean_squared_error', cv=k_fold)
gs2.fit(ds_boston.data, ds_boston.target)

# At what p the quality of cross-validation was optimal?
print("Optimal:", gs2.best_params_)
