# create model class with x and y FOR INPUT
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from selfmade_lime import lime_explainer
from lime.lime_tabular import LimeTabularExplainer


class ModelBuild:
    def __init__(self, model_type):
        if model_type == 'DecisionTree':
            self.dtree = DecisionTreeClassifier()
        else:
            print('default Decision Tree Classifier selected')
            self.dtree = DecisionTreeClassifier()

    def train_eval(self, X_train, y_train, X_test, y_test):
        self.dtree.fit(X_train, y_train)
        y_pred = self.dtree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return self.dtree

    @staticmethod
    def selfmade_explain(X_train):
        self_lime = lime_explainer.LimeTabularExplainer(training_data=np.array(X_train),
                                                        feature_names=X_train.columns,
                                                        class_names=['bad', 'good'],
                                                        mode='classification')
        return self_lime

    @staticmethod
    def explain(X_train):
        real_lime = LimeTabularExplainer(training_data=np.array(X_train),
                                         feature_names=X_train.columns,
                                         class_names=['bad', 'good'],
                                         mode='classification')
        return real_lime
