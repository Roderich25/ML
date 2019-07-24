import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def preprocessing(data):
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    data['Relatives'] = data.apply(lambda row: row['SibSp'] + row['Parch'], axis=1)
    data['Relatives'] = (data['Relatives'] - data['Relatives'].mean()) / data['Relatives'].std()
    data['Title'] = data['Name'].str.extract(r'(Miss|Mrs|Mr|Master)', expand=True)
    data['Title'] = data['Title'].fillna('Other')
    data['Title'] = data['Title'].map({'Master': 0, 'Miss': 1, 'Mrs': 2, 'Mr': 3, 'Other': 4}).astype(int)
    data['Age'] = data.groupby(['Title']).transform(lambda x: x.fillna(x.mean()))['Age']
    data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Fare'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()
    return data


df = pd.read_csv('train.csv')
X = preprocessing(df)

X = X[['Survived', 'Pclass', 'Sex', 'Age', 'Relatives', 'Title', 'Fare']]
X_train, X_test, y_train, y_test = train_test_split(X, X['Survived'], test_size=0.2, random_state=0)

X_train = X_train.drop(['Survived'], axis=1)
X_test = X_test.drop(['Survived'], axis=1)

clf = LogisticRegression(solver='liblinear')
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("LogisticRegression:", accuracy_score(y_test, predictions))

clf = MLPClassifier(solver='lbfgs', activation='tanh')
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("NeuralNetwork:", accuracy_score(y_test, predictions))

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("DecisionTree:", accuracy_score(y_test, predictions))

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Knn:", accuracy_score(y_test, predictions))

clf = RandomForestClassifier(n_estimators=10000, max_depth=10, random_state=0)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("RandomForest:", accuracy_score(y_test, predictions))

Kdf = pd.read_csv('test.csv')
Kaggle = preprocessing(Kdf)
output = pd.DataFrame()
output['PassengerId'] = Kaggle['PassengerId']
Kaggle = Kaggle[['Pclass', 'Sex', 'Age', 'Relatives', 'Title', 'Fare']]

predictions = clf.predict(Kaggle)
output['Survived'] = predictions
output.to_csv("mysubmission.csv", encoding='utf-8', index=False, header=True, columns=['PassengerId', 'Survived'])
