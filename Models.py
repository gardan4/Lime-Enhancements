# create model class with x and y FOR INPUT
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class Model():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2)

    def train(self):

        # random forest train on x and y
        self.dtree = DecisionTreeClassifier()
        self.dtree.fit(self.x, self.y)

    def predict(self):
        return self.dtree.predict(self.x_test)

    def evaluate(self):
        accuracy = accuracy_score(self.y_test, self.predict())
        class_report = classification_report(self.y_test, self.predict())
        cm = confusion_matrix(self.y, self.predict())
        return accuracy, class_report, cm
