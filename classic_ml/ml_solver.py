import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
import json
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.metrics import auc, precision_recall_curve
from collections import Counter


def pipeline():
    with open('config.json', 'r') as json_file:
        config_data = json.load(json_file)
    threshold_model = config_data['threshold_model']
    threshold_site = config_data['threshold_site']
    features_col = config_data['features_col']
    target_col = config_data['target_col']
    categorical_features = config_data['categorical_features']

    df_raw = create_features(threshold_model, threshold_site)
    x_train_res, y_train_res, x_val, y_val, x_test, y_test = prepare_dataset(df_raw, target_col, features_col)
    model = opt_catboost(x_train_res, x_val, y_train_res, y_val, categorical_features)
    testing_model(model, x_test, y_test)
    feature_importance_df = feature_importance_catboost(model, x_train_res, y_train_res, categorical_features)
    print('############################################')
    print('feature_importance info:')
    print(feature_importance_df)
    return


def feature_importance_catboost(trained_model, x_train: pd.DataFrame, y_train: pd.DataFrame,
                                categorical_features: list):
    train_pool = Pool(data=x_train, label=y_train, cat_features=categorical_features)
    feature_importance = trained_model.get_feature_importance(data=train_pool, type='FeatureImportance')
    feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df


def opt_catboost(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                 categorical_features: list):

    def objective(trial):
        params_cl = {
            "iterations": 300,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.6, log=True),
            "depth": trial.suggest_int("depth", 2, 11),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)
        }

        model_cl = CatBoostClassifier(**params_cl,
                                      cat_features=categorical_features,
                                      silent=True)
        model_cl.fit(x_train, y_train)
        y_predict_score = model_cl.predict_proba(x_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_predict_score)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)

        return auc_precision_recall

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    params = study.best_params
    params['iterations'] = 300
    params['cat_features'] = categorical_features
    params['eval_metric'] = 'PRAUC'

    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train, verbose=200)

    return model


def testing_model(model, x_test: pd.DataFrame, y_test: pd.Series):
    y_predict_score = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict_score)
    df = pd.DataFrame()
    df['precision'] = precision[:-1]
    df['recall'] = recall[:-1]
    df['threshold'] = thresholds
    df.to_csv('pr_recall_df.csv')
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)

    print(auc_precision_recall)
    return


def create_features(threshold_model, threshold_site) -> pd.DataFrame:
    df_x = pd.read_csv('data/interview.X.csv')
    df_y = pd.read_csv('data/interview.y.csv')
    df_x['reg_time'] = pd.to_datetime(df_x['reg_time'])
    # drop duplicates
    df_x.drop_duplicates(subset='uid', keep='last', inplace=True)
    df_x.fillna({'osName': 'unknown', 'model': 'unknown', 'hardware': 'unknown'}, inplace=True)

    df_y.drop_duplicates(subset='uid', keep='first', inplace=True)
    # merge df_x and df_y
    df = pd.merge(df_x, df_y, left_on='uid', right_on='uid', how='left')
    df.fillna({'tag': 'not_action'}, inplace=True)
    # add features about date
    df['dow'] = df['reg_time'].dt.dayofweek
    df['is_weekend'] = (df.dow > 4) * 1
    # change value if it less than threshold
    change_model = df['model'].value_counts().loc[lambda x: x < threshold_model].index
    df.loc[df['model'].isin(change_model), 'model'] = 'other'
    change_site = df['site_id'].value_counts().loc[lambda x: x < threshold_site].index
    df.loc[df['site_id'].isin(change_site), 'site_id'] = 'other'
    # features with hour
    df['time_hour'] = df['reg_time'].dt.hour
    df['time_minute'] = df['reg_time'].dt.minute
    df['time_minute'] = df['time_hour'] * 60 + df['time_minute']
    df['minute_sin'] = sin_transformer(24 * 60).fit_transform(df[['time_minute']])['time_minute']
    df['minute_cos'] = cos_transformer(24 * 60).fit_transform(df[['time_minute']])['time_minute']
    # add different target
    df['tag_any_action'] = 'not_action'
    df.loc[df['tag'] != 'not_action', 'tag_any_action'] = 'action'
    df['tag_dif_action'] = 'not_action'
    df.loc[df['tag'] == 'fclick', 'tag_dif_action'] = 'fclick'
    df.loc[(df['tag'] != 'fclick') & (df['tag'] != 'not_action'), 'tag_dif_action'] = 'view_through'
    df['tag_click'] = 'not_action'
    df.loc[df['tag'] == 'fclick', 'tag_click'] = 'fclick'

    return df


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def prepare_dataset(df_raw: pd.DataFrame, target_col: str, features_col: list):
    y = df_raw[target_col]
    y[y == 'not_action'] = 0
    if target_col == 'tag_dif_action':
        y[y == 'view_through'] = 2
        y[y == 'fclick'] = 1
    elif target_col == 'tag_any_action':
        y[y == 'action'] = 1
    elif target_col == 'tag_click':
        y[y == 'fclick'] = 1

    y = y.astype(np.int64)
    x = df_raw[features_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    sampling_strategy = 0.5
    if target_col == 'tag_dif_action':
        sampling_strategy = {2: len(y_train[y_train == 'view_through']),
                             1: len(y_train[y_train == 'fclick']),
                             0: len(y_train[y_train != 'not_action']) * 2}

    under_sample = RandomUnderSampler(sampling_strategy=sampling_strategy)
    x_train_res, y_train_res = under_sample.fit_resample(x_train, y_train)
    print(x_train.shape)
    print(x_train_res.shape)
    print(x_test.shape)
    print(x_val.shape)
    print(Counter(y_train))
    print(Counter(y_train_res))
    print(Counter(y_test))
    print(Counter(y_val))

    return x_train_res, y_train_res, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    pipeline()
