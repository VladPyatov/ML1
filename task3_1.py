import pandas as pd
from sklearn.svm import SVC

# Load the selection from the svm-data.csv file.
df = pd.read_csv('svm-data.csv', header=None)
X = df[[1, 2]].values
y = df[0].values
# Train the classifier with a linear kernel, parameter C = 100000 and random_state = 241.
svm = SVC(C=100000, kernel='linear', random_state=241)
svm.fit(X, y)
# Find the numbers of objects that are support (numbering from one).
print(svm.support_+1)

