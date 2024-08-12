import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt
from sklearn import tree
df = pd.read_csv('bank-full.csv', delimiter=';')  # Or 'bank.csv' if you used that one
print(df.head())
print(df.info())
print(df.describe())
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
df_encoded = pd.get_dummies(df, columns=categorical_cols)  # One-hot encoding for better handling of categorical data
X = df_encoded.drop('y', axis=1)  # Assuming 'y' is the target column for subscription (yes/no)
y = df_encoded['y']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5)  # Adjust max_depth as needed
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()
