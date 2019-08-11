# MachineLearning1_Week1
Dataset from https://www.kaggle.com/c/titanic/data
# Task 1:
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

# Task 2:
1. Download the selection from the titanic.csv file using the Pandas package.
2. Leave 4 features in the selection:
passenger class (Pclass), ticket price (Fare), passenger age (Age), and gender (Sex).
3. Note that the Sex feature has string values.
4. Select the target variable - it is recorded in the Survived column.
5. Find all objects that have NaN features and remove them from the selection.
6. Train the decision tree with the random_state = 241 parameter and other default parameters.
7. Calculate the importance of the features and find the two features with the greatest importance.
