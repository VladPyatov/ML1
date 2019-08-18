import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Download the training and test samples from perceptron-train.csv and perceptron-test.csv files.
df_train = pd.read_csv('perceptron-train.csv', header=None)
X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:,0].values

df_test = pd.read_csv('perceptron-test.csv', header=None)
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

# Train perceptron with standard parameters.
P = Perceptron()
P.fit(X_train, y_train)

# Calculate the accuracy of the resulting classifier in the test sample.
acc = accuracy_score(y_test, P.predict(X_test))

# Normalize training and test samples.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Train perceptron on a new sample.
P.fit(X_train_std, y_train)

# Calculate the accuracy of the resulting classifier in the (std) test sample.
acc_std = accuracy_score(y_test, P.predict(X_test_std))

# Find the difference between the accuracy in the test sample after normalization and the accuracy before it.
print(f"{acc_std-acc:.3f}")