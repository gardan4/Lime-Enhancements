from model_builder import ModelBuild
from sklearn.model_selection import train_test_split
from data_prep import DataPrep
import pandas as pd

data_location = "C:\\Users\\Tabea Heusel\\PycharmProjects\\Lime-Enhancements\\data\\compas-scores-two-years.csv"
compas_score_full = pd.read_csv(data_location)
data_prepper = DataPrep(compas_score_full)
X, y = data_prepper.get()
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=32, shuffle=True)
model = ModelBuild('DecisionTree')
model.train_eval(X_train, X_test, y_train, y_test)
exp = model.explain(X_train)
exp_inst = exp.explain_instance(X_train[1])
print("here")
exp_inst.save_to_file('visualizations/lime.html')
