from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split



Moons  =  make_moons(n_samples=10000, noise = 0.35, random_state= 42)

X  = Moons[0]
Y = Moons[1] 

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)




from Bagging import Bagging

baged_models = Bagging(model=LogisticRegression(), number_bags=4, sample_size=0.8, replacement=False)

if __name__ == '__main__':

    baged_models.fit(X_train,Y_train)

    y_pred = baged_models.predict(X_test)

    baged_models.accuracy(Y_test,y_pred)
