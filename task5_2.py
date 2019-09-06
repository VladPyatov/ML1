from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the samples from the gbm-data.csv file.
data = pd.read_csv('gbm-data.csv').values

# Split the samples into training and test.
X_train, X_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.8, random_state=241)

Min_Loss = []

# Train GradientBoostingClassifier (n_estimators = 250, verbose = True, random_state = 241).
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    gbc = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    gbc.fit(X_train, y_train)

    test_loss, train_loss = [], []

    # Use the staged_decision_function method to predict
    # the scores of the training and test samples at each iteration.
    # Transform the resulting prediction using the sigmoid function.
    for iter_ in gbc.staged_decision_function(X_train):
        train_loss.append(log_loss(y_train, [1.0/(1 + np.exp(-x)) for x in iter_]))

    for iter_ in gbc.staged_decision_function(X_test):
        test_loss.append(log_loss(y_test, [1.0/(1 + np.exp(-x)) for x in iter_]))

    Min_Loss.append((test_loss[np.argmin(test_loss)], np.argmin(test_loss)+1))

    # Calculate and plot the log-loss values on the training and test samples.
    plt.figure()
    plt.ylabel('log_loss')
    plt.xlabel('iteration')
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

# Give the minimum log-loss value on the test sample
# and the iteration number at which it is achieved with learning_rate = 0.2.
print(f"Min Loss = {Min_Loss[3][0]:.2f} at {Min_Loss[3][1]} iteration.")

# On the same data, train RandomForestClassifier with the number of trees equal
# to the number of iterations at which the best quality is achieved for the gradient boosting
# from the previous paragraph, with random_state = 241 and other default parameters.
rfc = RandomForestClassifier(n_estimators=37, random_state=241)
rfc.fit(X_train, y_train)

# What is the value of log-loss on the test for this random forest?
print(f"Log_loss = {log_loss(y_test, rfc.predict_proba(X_test)):.2f}")
