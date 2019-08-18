# Week1
Dataset from https://www.kaggle.com/c/titanic/data
# Task 1_1:
1. How many men and women were on the ship? As an answer, give two numbers separated by a space.
2. How many passengers survived? Calculate the proportion of surviving passengers.
Give the answer in percents
3. What is the proportion of first-class passengers among all passengers?
Give the answer in percents
4. How old were the passengers? Calculate the average and median age of passengers.
5. Do the number of brothers/sisters correlate with the number of parents/children?
Count the Pearson correlation between traits SibSp и Parch.
6. What is the most popular female name on the ship?
Extract the passenger’s full name (Name column) from his personal name (First Name).

# Task 1_2:
1. Download the selection from the titanic.csv file using the Pandas package.
2. Leave 4 features in the selection:
passenger class (Pclass), ticket price (Fare), passenger age (Age), and gender (Sex).
3. Note that the Sex feature has string values.
4. Select the target variable - it is recorded in the Survived column.
5. Find all objects that have NaN features and remove them from the selection.
6. Train the decision tree with the random_state = 241 parameter and other default parameters.
7. Calculate the importance of the features and find the two features with the greatest importance.

# Week2
# Task 2_1:
1. Download the Wine set.
2. Extract features and classes from the data.
3. Create a splitter generator
4. Find the cross-validation classification accuracy for the k nearest neighbors method for k from 1 to 50.
5. Scale features.
6. What value of k turned out to be optimal after reducing the features to the same scale?
   Did feature scaling help?

# Task 2_2:
1. Download the Boston set.
2. Scale features.
3. Create a splitter generator.
4. Go through the different options for the metric parameter p over the grid
   from 1 to 10 in such a way that 200 options are tested in total.
5. At what p the quality of cross-validation was optimal?

# Task 2_3:
1. Download the training and test samples from perceptron-train.csv and perceptron-test.csv files.
2. Train perceptron with standard parameters.
3. Calculate the accuracy of the resulting classifier in the test sample.
4. Normalize training and test samples.
5. Train perceptron on a new sample.
6. Calculate the accuracy of the resulting classifier in the (std) test sample.
7. Find the difference between the accuracy in the test sample after normalization and the accuracy before it.

