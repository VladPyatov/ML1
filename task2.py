import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Download the selection from the titanic.csv file using the Pandas package.
df = pd.read_csv('titanic.csv', index_col='PassengerId')

# Leave 4 features in the selection:
# passenger class (Pclass), ticket price (Fare), passenger age (Age), and gender (Sex).
X = df.loc[:, ['Pclass', 'Fare', 'Age', 'Sex']]

# Note that the Sex feature has string values.
X['Sex'] = np.where(X['Sex'] == 'male', 1, 0)

# Select the target variable - it is recorded in the Survived column.
Y = df['Survived']
# Find all objects that have NaN features and remove them from the selection.
X = X.dropna()
Y = Y[X.index.values]

# Train the decision tree with the random_state = 241 parameter and other default parameters.
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

# Calculate the importance of the features and find the two features with the greatest importance.
important = clf.feature_importances_
indices = np.argsort(important)[::-1]
for i in range(len(X.columns)):
    print(f"{i}){X.columns[indices[i]]} = {important[indices[i]]}")

print("Most important:", X.columns[indices[0]], ',', X.columns[indices[1]])

