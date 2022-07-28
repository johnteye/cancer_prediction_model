
#importing the python module
import sklearn
#building the model
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
#importing the datasets
from sklearn.datasets import load_breast_cancer

#importing the accuracy measuring function
from sklearn.metrics import accuracy_score

 #loading the datasets
data = load_breast_cancer()

#organizing our data
label_names = data['target_names']
labels =data['target']
feature_names = data['feature_names']
features = data['data']

#looking at the data
print(label_names)

#splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

#initializing the classifer
gnb = GaussianNB()

#training the classifer
model = gnb.fit(train, train_labels)

#making the predictions
predictions = gnb.predict(test)
print(predictions)

#evaluating the accuracy
print(f'the accuracy is {accuracy_score(test_labels, predictions)}')


