import pandas as pd
import numpy as np

class DataPrep:
    def __init__(self, data):
        self.data = data

    def get(self):
        compas_df = self.data

        compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                                  (compas_df['days_b_screening_arrest'] >= -30) &
                                  (compas_df['is_recid'] != -1) &
                                  (compas_df['c_charge_degree'] != "O") &
                                  (compas_df['score_text'] != "NA")]

        compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
        X = compas_df[['age', 'two_year_recid', 'c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

        y = np.array([0 if score == 'High' else 1 for score in compas_df['score_text']])
        sens = X.pop('race')

        # assign African-American as the protected class
        X = pd.get_dummies(X)
        sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
        X['race'] = sensitive_attr

        # make sure everything is lining up
        assert all((sens == 'African-American') == (X['race'] == 1))
        cols = [col for col in X]

        return X, y, cols
