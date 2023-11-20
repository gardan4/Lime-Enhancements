from model_builder import ModelBuild
from sklearn.model_selection import train_test_split
from data_prep import DataPrep
import pandas as pd
if __name__ == 'main':
    compas_score_full = pd.read_csv('data/compas-scores-two-years.csv')
    data_prepper = DataPrep(compas_score_full)
    X, y = data_prepper.get()
    print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=32, shuffle=True)
    model = ModelBuild('DecisionTree')
    model.train_eval(X_train, X_test, y_train, y_test)
