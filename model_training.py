import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle

data = pd.read_csv('student_mat_2173a47420.csv', sep=';')
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print(accuracy)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])