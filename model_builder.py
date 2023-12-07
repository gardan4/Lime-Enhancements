# create model class with x and y FOR INPUT
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from somlime.lime_tabular import LimeTabularExplainerSOM
from lime.lime_tabular import LimeTabularExplainer


class ModelBuild:
    def __init__(self, model_type):
        if model_type == 'DecisionTree':
            self.model = DecisionTreeClassifier()
        elif model_type == 'Logistic':
            print('logistic regression selected')
            self.model = LogisticRegression()
        else:
            print('default Decision Tree Classifier selected')
            self.model = DecisionTreeClassifier()

    def train_eval(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return self.model, accuracy, cm

    @staticmethod
    def som_explain(X_train, X_cols):
        self_lime = LimeTabularExplainerSOM(training_data=X_train,
                                            feature_names=X_cols,
                                            class_names=['bad', 'good'],
                                            mode='classification')
        return self_lime

    @staticmethod
    def explain(X_train, X_cols):
        real_lime = LimeTabularExplainer(training_data=X_train,
                                         feature_names=X_cols,
                                         class_names=['bad', 'good'],
                                         mode='classification')
        return real_lime
