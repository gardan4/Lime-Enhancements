
class DataPrep:
    def __init__(self, data):
        self.data = data

    def get(self):
        cols = ['sex', 'age', 'age_cat', 'race', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'c_jail_in',
                'c_jail_out', 'c_charge_degree', 'is_recid', 'score_text', 'priors_count.1', 'two_year_recid']
        compas_score = self.data[cols]

        compas_score = compas_score[compas_score['days_b_screening_arrest'] <= 30]
        compas_score = compas_score[compas_score['days_b_screening_arrest'] >= -30]
        compas_score = compas_score[compas_score['is_recid'] != -1]
        compas_score = compas_score[compas_score['c_charge_degree'] != 'O']
        compas_score = compas_score[compas_score['score_text'] != 'N/A']
        X = compas_score[['days_in_jail', 'age', 'sex', 'decile_score', 'priors_count', 'c_days_from_compas', 'is_violent_recid',
             'v_decile_score']]
        y = compas_score['is_recid']
        return X, y
