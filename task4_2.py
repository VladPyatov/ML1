import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Load data from close_prices.csv file.
df = pd.read_csv('close_prices.csv')

# On the downloaded data, train the PCA with 10 components.
pca = PCA(n_components=10)
X = pca.fit_transform(df.iloc[:, 1:])

# How many components are enough to explain 90% of the variance?
i = 1
while np.sum(pca.explained_variance_ratio_[0:i])*100 < 90:
    i += 1
print(i)

# Load the Dow Jones Index information from djia_index.csv file.
dj = pd.read_csv('djia_index.csv')

# What is the Pearson correlation between the first component and the Dow Jones index?
corr = np.corrcoef(X[:, 0], dj['^DJI'])
print(corr)

# Which company has the most weight in the first component?
name = df.columns[np.argmax(pca.components_[0])+1]
print(name)

