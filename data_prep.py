import pandas as pd

class DataPrep:
    def __init__(self, data):
        self.data = data

    def get(self):
        compas_score = self.data
        compas_score["sex"].replace({'Male': 1, 'Female': 0}, inplace=True)
        compas_score["is_recid"].replace({0: False, 1: True}, inplace=True)
        compas_score = compas_score.drop(
            ['last', 'first', 'out_custody', 'in_custody', 'c_offense_date', 'decile_score.1', 'priors_count.1',
             'c_case_number', 'start', 'end', 'event', 'screening_date', 'c_case_number',
             'juv_other_count', 'juv_misd_count', 'juv_fel_count', 'r_days_from_arrest', 'id', 'r_charge_degree',
             'r_offense_date', 'vr_case_number', 'r_case_number', 'r_jail_out', 'c_arrest_date', 'r_charge_desc',
             'r_jail_in', 'violent_recid', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc'], axis=1)
        compas_score['c_jail_in'] = pd.to_datetime(compas_score['c_jail_in'])
        compas_score['c_jail_out'] = pd.to_datetime(compas_score['c_jail_out'])
        compas_score['days_in_jail'] = abs((compas_score['c_jail_out'] - compas_score['c_jail_in']).dt.days)
        # TODO: mean modeling should only be done after train test splitting not before
        for col in compas_score.columns:
            if compas_score[col].dtype == "object":
                compas_score[col] = compas_score[col].fillna("UNKNOWN")
            else:
                compas_score[col] = compas_score[col].fillna(compas_score[col].mean())

        compas_score = compas_score[compas_score['days_b_screening_arrest'] <= 30]
        compas_score = compas_score[compas_score['days_b_screening_arrest'] >= -30]
        compas_score = compas_score[compas_score['is_recid'] != -1]
        compas_score = compas_score[compas_score['c_charge_degree'] != 'O']
        compas_score = compas_score[compas_score['score_text'] != 'N/A']
        #get all the unique values from race
        compas_score['race'].replace({'Other':0, 'African-American':1, 'Caucasian':2, 'Hispanic':3, 'Asian':4, 'Native American':5}, inplace=True)



        # X = compas_score[['days_in_jail', 'age', 'decile_score', 'priors_count', 'c_days_from_compas', 'is_violent_recid','v_decile_score']]
        X = compas_score[['days_in_jail', 'age', 'decile_score', 'priors_count', 'c_days_from_compas', 'race', 'v_decile_score']]
        y = compas_score['is_recid']
        return X, y
