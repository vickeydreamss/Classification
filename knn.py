#K Nearest Neighbour is a simple algorithm that stores all the available cases and classifies the new data or case based on a similarity measure

#Data set :  car evualation data set from UCI machine learning repository

# loading dependencies

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

#print(data.head())

# converting non-numerical data into numerical form. the algo can do lot better with numerical data
# sometime data may look numerical,but if you check the type, it might still be strings
# we can use fit_transfrom fromlabel encoder to transform non-numerical data to numerical data


lab_encoder = preprocessing.LabelEncoder()
buying = lab_encoder.fit_transform(list(data["buying"]))
maint = lab_encoder.fit_transform(list(data["maint"]))
door = lab_encoder.fit_transform(list(data["door"]))
persons = lab_encoder.fit_transform(list(data["persons"]))
lug_boot = lab_encoder.fit_transform(list(data["lug_boot"]))
safety = lab_encoder.fit_transform(list(data["safety"]))
cls = lab_encoder.fit_transform(list(data["class"]))

#print(buying, maint,door,safety)

# x is going to be our features and y is going to be target (label)
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# k is basically hom many parameters we want, and k is a hyperparameter means we can tweak as and when we progress
# alway keep k to an odd number (3, 5, 7...), don't try with even number
# alog calculates euclidean distance and finds out nearest neighbours

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]],9, True)
    print("N: ", n)











