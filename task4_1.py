import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

# Load job descriptions and corresponding annual salaries from salary-train.csv file.
data_train = pd.read_csv('salary-train.csv')

# Bring text to lower case.
data_train['FullDescription'] = data_train['FullDescription'].map(lambda x: x.lower())

# Replace everything except letters and numbers with spaces.
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

# Use TfidfVectorizer to convert texts to feature vectors.
vectorizer = TfidfVectorizer(min_df=5)
X_train_text = vectorizer.fit_transform(data_train['FullDescription'])

# Replace the omissions in the LocationNormalized and ContractTime columns with the special string 'nan'.
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

# Use the DictVectorizer to get one-hot coding of the LocationNormalized and ContractTime features.
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Combine all the received features in one matrix "objects-features".
X_train = hstack((X_train_text, X_train_categ))
y_train = data_train['SalaryNormalized'].values

# Train ridge regression with alpha = 1 and random_state = 241.
regr = Ridge(alpha=1, random_state=241)
regr.fit(X_train, y_train)

# Build predictions for two examples from the salary-test-mini.csv file.
data_test = pd.read_csv('salary-test-mini.csv')

data_test['FullDescription'] = data_test['FullDescription'].map(lambda x: x.lower())
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

X_test_text = vectorizer.transform(data_test['FullDescription'])

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test = hstack((X_test_text, X_test_categ))

y_test = regr.predict(X_test)

print(y_test)