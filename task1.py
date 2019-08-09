#Воронцов, МО

import pandas as pd
import re

data = pd.read_csv('titanic.csv', index_col='PassengerId')

# 1. How many men and women were on the ship? As an answer, give two numbers separated by a space.
sex_counts = data['Sex'].value_counts()
print(f"{sex_counts['male']} {sex_counts['female']}")

# 2. How many passengers survived? Calculate the proportion of surviving passengers.
# Give the answer in percents

Pass = data['Survived'].value_counts()
Percent = Pass[1] / Pass.sum() * 100
print(f"{Percent:0.2f}")

# 3. What is the proportion of first-class passengers among all passengers?
# Give the answer in percents

All = data['Pclass'].value_counts()
Percent = All[1] / All.sum() * 100
print(f"{Percent:0.2f}")

# 4. How old were the passengers? Calculate the average and median age of passengers.

print(f"{data['Age'].mean():0.2f} {data['Age'].median():0.2f}")

# 5. Do the number of brothers/sisters correlate with the number of parents/children?
# Count the Pearson correlation between traits SibSp и Parch.

corr = data['SibSp'].corr(data['Parch'])
print(f"{corr:0.2f}")


# 6. What is the most popular female name on the ship?
# Extract the passenger’s full name (Name column) from his personal name (First Name).

def Extr_Name(name):
    search = re.search('^[^,]+, (.*)', name, flags=re.MULTILINE)
    if search:
        name = search[1]

    search = re.search('\(([^)]+)\)', name)
    if search:
        name = search[1]

    name = re.sub('Miss\. |Mrs\. |Ms\.', '', name)
    name = name.split(' ')[0]
    return name

name = data[data['Sex'] == 'female']['Name'].map(Extr_Name).value_counts()

print(name.head(1).index[0])
