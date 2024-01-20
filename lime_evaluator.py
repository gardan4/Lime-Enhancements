import pandas as pd
from model_builder import ModelBuild
from scipy.stats import f_oneway
#disable pandas future warning
pd.options.mode.chained_assignment = None  # default='warn'

class Lime_eval:
    @staticmethod
    def evaluate_stability(inst, model_prob, explainer_lime, som_model=0,  instances=50, experiment=0):
        # Explaination stability test
        explainations = []
        if experiment == 0:
            for i in range(instances):
                exp_inst = explainer_lime.explain_instance(inst, model_prob, num_features=7)
                explainations += [exp_inst.as_list()]
        else:
            for i in range(instances):
                exp_inst = explainer_lime.explain_instance(inst, model_prob, som_model, num_features=7, plot=False, experiment=3)
                explainations += [exp_inst.as_list()]

        # Extracting unique categories
        categories = sorted({item[0] for sublist in explainations for item in sublist})

        # Creating DataFrame
        df = pd.DataFrame(columns=categories)
        # Populating the DataFrame
        for sublist in explainations:
            # Initialize a row with all None values
            row = {cat: None for cat in categories}
            # Update the values for existing categories
            for category, value in sublist:
                row[category] = value
            # Append the row to the DataFrame
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

        return df

    @staticmethod
    def compare_experiments(df_exp0, df_exp1, p_value_cutoff=0.05, all=True, mean=True):
        """
        Compares the variances of two experiments using F-test.

        Parameters:
        df_exp0 (pd.DataFrame): DataFrame containing data from the original experiment.
        df_exp1 (pd.DataFrame): DataFrame containing data from the modified experiment.
        p_value_cutoff (float): The threshold for determining statistical significance.

        Returns:
        None
        """
        if all:
            significant_results = {}
            non_significant_results = {}

            for column in df_exp0.columns:
                f_statistic, p_value = f_oneway(df_exp0[column], df_exp1[column])
                variance_relation = "HIGHER" if df_exp1[column].var() > df_exp0[column].var() else "LOWER"

                # Formatting the F-statistic and p-value to 4 decimal places
                f_statistic_formatted = f"{f_statistic:.4f}"
                p_value_formatted = f"{p_value:.6f}"

                if p_value < p_value_cutoff:
                    significant_results[column] = (f_statistic_formatted, p_value_formatted, variance_relation)
                else:
                    non_significant_results[column] = (f_statistic_formatted, p_value_formatted, variance_relation)

            print("Experiment all columns:")
            print("-------------------------")
            # Print the significant results
            print("Significant Results:")
            for variable, (f_stat, p_val, var_rel) in significant_results.items():
                print(
                    f"{variable:35} F-statistic = {f_stat:10} p-value = {p_val:10} Variance is {var_rel} in modified experiment")

            # Print a separator for clarity
            print("\nNon-Significant Results:")
            # Print the non-significant results
            for variable, (f_stat, p_val, var_rel) in non_significant_results.items():
                print(
                    f"{variable:35} F-statistic = {f_stat:10} p-value = {p_val:10} Variance is {var_rel} in modified experiment")

        if mean:
            df0mean = (df_exp0 ** 2).mean(axis=1)
            df0mean = pd.DataFrame(df0mean, columns=['mean'])

            df1mean = (df_exp1 ** 2).mean(axis=1)
            df1mean = pd.DataFrame(df1mean, columns=['mean'])

            significant_results = {}
            non_significant_results = {}

            for column in df0mean.columns:
                f_statistic, p_value = f_oneway(df0mean[column], df1mean[column])
                variance_relation = "HIGHER" if df1mean[column].var() > df0mean[column].var() else "LOWER"

                # Formatting the F-statistic and p-value to 4 decimal places
                f_statistic_formatted = f"{f_statistic:.4f}"
                p_value_formatted = f"{p_value:.6f}"

                if p_value < p_value_cutoff:
                    significant_results[column] = (f_statistic_formatted, p_value_formatted, variance_relation)
                else:
                    non_significant_results[column] = (f_statistic_formatted, p_value_formatted, variance_relation)

            print("\nExperiment squared mean")
            print("-------------------------")
            # Print the significant results
            print("Significant Results:")
            for variable, (f_stat, p_val, var_rel) in significant_results.items():
                print(
                    f"{variable:35} F-statistic = {f_stat:10} p-value = {p_val:10} Variance is {var_rel} in modified experiment")

            # Print a separator for clarity
            print("\nNon-Significant Results:")
            # Print the non-significant results
            for variable, (f_stat, p_val, var_rel) in non_significant_results.items():
                print(
                    f"{variable:35} F-statistic = {f_stat:10} p-value = {p_val:10} Variance is {var_rel} in modified experiment")