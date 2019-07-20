import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv', dtype=np.uint8)

features = [f"pixel{i}" for i in range(0, 28 * 28)]

X_train, X_test, y_train, y_test = train_test_split(df[features], df['label'], test_size=0.20, random_state=0)

pca = IncrementalPCA(n_components=330)
pca.fit(X_train)
print(pca.n_components_)
print(np.sum(pca.explained_variance_ratio_))

pca_train = pca.transform(X_train)
pca_test = pca.transform(X_test)

clf = MLPClassifier(solver='lbfgs', activation='tanh')
clf = clf.fit(pca_train, y_train)
print("Neural Network:", accuracy_score(y_test, clf.predict(pca_test)))

clf = KNeighborsClassifier(n_neighbors=10)
clf = clf.fit(pca_train, y_train)
print("KNN:", accuracy_score(y_test, clf.predict(pca_test)))

pca.fit(df[features])
pca_all = pca.transform(df[features])
clf.fit(pca_all, df['label'])

digits = pd.read_csv('test.csv', dtype=np.uint8)
pca_digits = pca.transform(digits)
predictions = clf.predict(pca_digits)

output = pd.DataFrame()
output['ImageId'] = range(1, 28001)
output['Label'] = predictions
output.to_csv("mysubmission.csv", encoding='utf-8', index=False, header=True, columns=['ImageId', 'Label'])
