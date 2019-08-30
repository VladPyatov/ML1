import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Download objects from the news dataset 20 newsgroups related to the cosmos and atheism categories.
newgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
y = newgroups.target
X = newgroups.data
# Calculate TF-IDF features for all texts.
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)
names = tfidf.get_feature_names()

# Find the best C parameter for SVM (kernel = 'linear') using cross-validation over 5 blocks.
cv = KFold(n_splits=5, shuffle=True, random_state=241)
svm = SVC(random_state=241,kernel='linear')
param_grid = {'C': np.power(10.0, np.arange(-5, 6))}
gs = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

# Train SVM throughout the sample with the optimal C parameter found in the previous step.
clf = gs.best_estimator_
clf.fit(X, y)

# Find the 10 words with the highest absolute weight.
index = np.argsort(np.abs(clf.coef_.toarray()[0]))[-10::1]
top_10 = sorted(map(lambda i: names[i], index))
print(top_10)
