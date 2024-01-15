import pandas as pd
#disable pandas future warning
pd.options.mode.chained_assignment = None  # default='warn'
class Lime_eval:
    @staticmethod
    def evaluate_stability(exp_inst, instances=30):
        # Explaination stability test
        explainations = []

        for i in range(instances):
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

        # get the mean and std for each category and round them to 2 decimals
        return df

        # get the average of the mean and std
        # (df.describe().mean(axis=1))