import pandas as pd


class Lime_eval:
    @staticmethod
    def evaluate_stability(exp_inst, instances=30):
        # Explanation stability test
        explanations = []

        for i in range(instances):
            explanations += [exp_inst.as_list()]

        # Extracting unique categories
        categories = sorted({item[0] for sublist in explanations for item in sublist})

        # Creating DataFrame
        df = pd.DataFrame(columns=categories)
        # Populating the DataFrame
        for sublist in explanations:
            # Initialize a row with all None values
            row = {cat: None for cat in categories}
            # Update the values for existing categories
            for category, value in sublist:
                row[category] = value
            # Append the row to the DataFrame
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

        # get the mean and std for each category
        print(df.describe().loc[['mean', 'std']])

        # get the average of the mean and std
        print(df.describe().mean(axis=1))
