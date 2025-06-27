import fire
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import os
import pickle

class TrainEvaluateModels:
    _output_path_train = ""
    _output_path_preprocess = ""
    _name_path = ""

    def _create_output_path_train(self):
        if not(os.path.exists(self._output_path_train)):
            os.mkdir(self._output_path_train)
        if not(os.path.exists(f'{self._output_path_train}/feature_importance')):
            os.mkdir(f'{self._output_path_train}/feature_importance')
        if not(os.path.exists(f'{self._output_path_train}/feature_importance/{self._name_path}')):
            os.mkdir(f'{self._output_path_train}/feature_importance/{self._name_path}')
        if not(os.path.exists(f'{self._output_path_train}/models')):
            os.mkdir(f'{self._output_path_train}/models')
        if not(os.path.exists(f'{self._output_path_train}/models/{self._name_path}')):
            os.mkdir(f'{self._output_path_train}/models/{self._name_path}')
        if not(os.path.exists(f'{self._output_path_train}/metrics')):
            os.mkdir(f'{self._output_path_train}/metrics')
        if not(os.path.exists(f'{self._output_path_train}/metrics/{self._name_path}')):
            os.mkdir(f'{self._output_path_train}/metrics/{self._name_path}')

    def _get_preprocess_x_columns(self):
        x_cols = pd.read_csv(f'{self._output_path_preprocess}/x_col_names.csv')['x_col'].to_list()
        return x_cols

    def _get_preprocess_y_column(self):
        y_col = pd.read_csv(f'{self._output_path_preprocess}/y_col_name.csv')['y_col'].to_list()
        return y_col


    def __init__(self, output_path_train, output_path_preprocess, name_path):
        self._output_path_train = output_path_train
        self._output_path_preprocess = output_path_preprocess
        self._name_path = name_path
        self._create_output_path_train()

    def train_evaluate_models(self, df_data_train, model_parameters_grid, model):
        x_cols = self._get_preprocess_x_columns()
        y_col = self._get_preprocess_y_column()
        grid_search = GridSearchCV(estimator=model, param_grid=model_parameters_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(df_data_train[x_cols], df_data_train[y_col].values.ravel())

        df_model_results = pd.DataFrame({'model_parameters': grid_search.cv_results_['params'],
                                         'model_rank': grid_search.cv_results_['rank_test_score'],
                                         'auc_score_mean': grid_search.cv_results_['mean_test_score'],
                                         'auc_score_std': grid_search.cv_results_['std_test_score']})
        df_model_results['auc_score_cv'] = df_model_results['auc_score_std'] / df_model_results['auc_score_mean']
        df_model_results.to_csv(f'{self._output_path_train}/metrics/{self._name_path}/train_cv_model_results.csv', index=False)

        df_model_results_best_model = df_model_results[df_model_results['model_rank']==1]
        df_model_results_best_model.to_csv(f'{self._output_path_train}/metrics/{self._name_path}/train_cv_model_results_best_model.csv', index=False)

        df_feature_importance = pd.DataFrame({'variable': grid_search.feature_names_in_, 'importance': grid_search.best_estimator_.feature_importances_})
        df_feature_importance.to_csv(f'{self._output_path_train}/feature_importance/{self._name_path}/feature_importance.csv', index=False)

        with open(f'{self._output_path_train}/models/{self._name_path}/grid_search_model.pickle', 'wb') as handle:
            pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def process_train_evaluate_models(model_parameters_grid, model, name_path):
    if (os.getcwd().endswith("src")):
        os.chdir("..")
    df_data_train = pd.read_csv("data/out/data_train_prepared.csv")
    train_validate_models_instance = TrainEvaluateModels(output_path_train="outputs/train", output_path_preprocess="outputs/preprocess", name_path=name_path)
    train_validate_models_instance.train_evaluate_models(df_data_train, model_parameters_grid, model)

def main():
    model_parameters_grid_xgbost = {'max_depth': [3, 5, 7],'eta': [0.05, 0.1, 0.2], 'gamma': [0, 1, 5], 'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0], 'n_estimators': [50, 100], 'scale_pos_weight': [1, 5, 10]}
    xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='auc',random_state=42)
    model_parameters_grid_random_forest = {'n_estimators': [50, 100, 200], 'max_depth': [None, 4, 6, 8],              
    'min_samples_leaf': [1, 10, 50], 'min_impurity_decrease': [0.0, 0.01, 0.05]}
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    process_train_evaluate_models(model_parameters_grid_xgbost,xgb_model,'xgbost')
    process_train_evaluate_models(model_parameters_grid_random_forest,rf_model, 'random_forest')

if __name__ == "__main__":
    fire.Fire(main)
