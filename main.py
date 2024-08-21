import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

df = sns.load_dataset('iris')
df

x = df.iloc[:, :-1]
y = df['species']

# split train test from x, y test size = 0.2
# use k-nearest neighbors to classify x_train, y_train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create sidebar menu for user can select classifiers
classifier = st.sidebar.selectbox('Classifier', ('KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Neural Network'))

if classifier == 'KNN':
  knn = KNeighborsClassifier(n_neighbors=3) # 20% of maximum train number
  knn.fit(x_train, y_train)
  # use SVM to classify x_train, y_train
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  accuracy_score(y_test, y_pred)

if classifier == 'SVM':
  # use dicision tree to classify x_train, y_train
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  y_pred = dt.predict(x_test)
  accuracy_score(y_test, y_pred)

if classifier == 'Random Forest':
# use random forest to classify x_train, y_train
  rf = RandomForestClassifier()
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  accuracy_score(y_test, y_pred)

if classifier == 'Neural Network':
# use NN to classify x_train, y_train
  nn = MLPClassifier()
  nn.fit(x_train, y_train)
  y_pred = nn.predict(x_test)
  accuracy_score(y_test, y_pred)



