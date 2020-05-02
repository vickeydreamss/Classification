# support vector machine algo

import sklearn
from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np

cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)


x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.15)

#print(x_train)
#print(y_test)

classes = ['malignant' 'benign']