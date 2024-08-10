import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df = pd.read_csv('bank.csv')
print(df.head())
print(df.info())
print(df.describe())

missing_values = df.isnull().sum()
print(missing_values)
df = df.drop_duplicates()

le = LabelEncoder()
df['job'] = le.fit_transform(df['job'])
df['marital'] = le.fit_transform(df['marital'])
df['education'] = le.fit_transform(df['education'])
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['month'] = le.fit_transform(df['month'])
df['poutcome'] = le.fit_transform(df['poutcome'])
df['y'] = le.fit_transform(df['bank.csv']) # Target variable

X = df.drop('y', axis=1)
y = df['y']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=10)
clf1 = clf.fit(X_train,y_train)
y_pred = clf1.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from matplotlib import pyplot as plt
from sklearn import tree

fig = plt.figure()
 



