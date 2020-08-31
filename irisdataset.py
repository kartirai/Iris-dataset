# iris dataset
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('/home/kartikey/Downloads/iris.csv')
data.head(5)

data.describe()

data['Species'].value_counts()

tmp = data.drop(['Id'],axis=1)
g = sns.pairplot(tmp,hue='Species',markers='*')
plt.show()

#model

X = data.drop(['Id','Species'],axis=1)
y = data['Species']
print(X.shape)
print(y.shape)

#splitting data into test and train data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(k_range,scores)
plt.xlabel('Value of k in KNN')
plt.ylabel('Accuracy score')
plt.title('Accuracy scores for values of k of k-nearest-neighbors')
plt.show()

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X,y)
knn.predict([[6,3,4,2]])

