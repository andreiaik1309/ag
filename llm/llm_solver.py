import interpreter
import json

interpreter.api_key = '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'

with open('config.json', 'r') as json_file:
    config_data = json.load(json_file)

name_file_train = config_data['name_file_train']
name_file_test = config_data['name_file_test']
name_file_val = config_data['name_file_val']
features_col = config_data['features_col']
categorical_features = config_data['categorical_features']
target_col = config_data['target_col']

base_prompt = """Identify Important Factors: Find out what factors are most likely to make someone
            click on online ads. Tell me which factors are the strongest predictors and rank them"""


total_prompt = f"""File '{name_file_train}' contains data with the ad display logins  for training the model.
                File '{name_file_test}' contains data with the ad display logins for testing the model.
                File '{name_file_val}' contains data with the ad display logins for validating the model (finding
                optimal hyper parameter for the model).
                The column '{target_col}' contains the target variable, whether the click occurred or not.
                Columns {features_col} contain features (signs that affect predictions whether a banner 
                with an advertisement will be clicked on or not), including the {categorical_features} columns
                are categorical.
                Plan:
                necessary to train a model for predicting clicks on an advertising banner (use gradient boosting models
                 on trees, for example, CatBoost); 
                It is very important to keep in mind that in the test and validation dataset, the classes are very 
                unbalanced, and in the training classes they are balanced. Therefore, use the precision-recall AUC
                 metric as a model quality metric for test and validation datasets.
                for it write python code that will include
                1. preparing the datasets;
                2. train model; 
                3. testing the model;
                4. calculate all the metrics for classification tasks (including 
                PR-AUC Precision-Recall Area Under the Curve) on the test dataset.        

                In the end, you have to answer the main question {base_prompt}.
                

            """

interpreter.chat(total_prompt)
