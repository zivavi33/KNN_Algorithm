from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from KNN_classifier import Knn_algorithm

# load the data set 
data = load_iris()
x, y  = data.data , data.target
# split to train and test
#? set random seed for tuining hyper parameters so the split will be consistent and the different in the results will be results of your tuining.
X_train,X_test,Y_train,Y_test = train_test_split(x, y,test_size = 0.25,random_state = 42)

model = Knn_algorithm(k = 3)
model.fit(X_train,Y_train)
Y_predicted = model.predict(X_test)
acc = model.accuracy(Y_test,Y_predicted)
print(f" accuracy of knn model: {acc:.2%}.")