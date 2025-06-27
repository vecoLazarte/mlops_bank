import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import os
import pickle
import mlflow
from mlflow.artifacts import download_artifacts

class TrainEvaluateModels:
    _output_path_train = ""
    _output_path_preprocess = ""
    _name_path = ""
    _preprocess_train_run_id = ""

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
        x_cols_path = download_artifacts(run_id=self._preprocess_train_run_id, artifact_path=f'{self._output_path_preprocess}/x_col_names.csv')
        x_cols = pd.read_csv(x_cols_path)['x_col'].to_list()
        return x_cols

    def _get_preprocess_y_column(self):
        y_col_path = download_artifacts(run_id=self._preprocess_train_run_id, artifact_path=f'{self._output_path_preprocess}/y_col_name.csv')
        y_col = pd.read_csv(y_col_path)['y_col'].to_list()
        return y_col


    def __init__(self, output_path_train, output_path_preprocess, name_path, preprocess_train_run_id):
        self._output_path_train = output_path_train
        self._output_path_preprocess = output_path_preprocess
        self._name_path = name_path
        self._preprocess_train_run_id = preprocess_train_run_id
        self._create_output_path_train()

    def train_evaluate_models(self, df_data_train, model_parameters_grid, model):
        x_cols = self._get_preprocess_x_columns()
        y_col = self._get_preprocess_y_column()
        mlflow.log_param(f"train_rows", df_data_train.shape[0])
        mlflow.log_param(f"train_cols", x_cols)
        grid_search = GridSearchCV(estimator=model, param_grid=model_parameters_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(df_data_train[x_cols], df_data_train[y_col].values.ravel())

        prefixed_params = {f"best_param_{self._name_path}_{k}": v for k, v in grid_search.best_params_.items()}
        mlflow.log_params(prefixed_params)

        df_model_results = pd.DataFrame({'model_parameters': grid_search.cv_results_['params'],
                                         'model_rank': grid_search.cv_results_['rank_test_score'],
                                         'auc_score_mean': grid_search.cv_results_['mean_test_score'],
                                         'auc_score_std': grid_search.cv_results_['std_test_score']})
        df_model_results['auc_score_cv'] = df_model_results['auc_score_std'] / df_model_results['auc_score_mean']
        
        df_model_results.to_csv(f'{self._output_path_train}/metrics/{self._name_path}/train_cv_model_results.csv', index=False)
        mlflow.log_artifact(f'{self._output_path_train}/metrics/{self._name_path}/train_cv_model_results.csv', artifact_path=f'{self._output_path_train}/metrics/{self._name_path}')

        df_model_results_best_model = df_model_results[df_model_results['model_rank']==1]

        best_auc_score_mean = df_model_results_best_model['auc_score_mean'].values[0]
        mlflow.log_metric(f"best_cv_train_{self._name_path}_auc_score_mean", best_auc_score_mean)
        best_auc_score_std = df_model_results_best_model['auc_score_std'].values[0]
        mlflow.log_metric(f"best_cv_train_{self._name_path}_auc_score_std", best_auc_score_std)
        best_auc_score_cv = df_model_results_best_model['auc_score_cv'].values[0]
        mlflow.log_metric(f"best_cv_train_{self._name_path}_auc_score_cv", best_auc_score_cv)

        df_model_results_best_model.to_csv(f'{self._output_path_train}/metrics/{self._name_path}/train_cv_model_results_best_model.csv', index=False)
        mlflow.log_artifact(f'{self._output_path_train}/metrics/{self._name_path}/train_cv_model_results_best_model.csv', artifact_path=f'{self._output_path_train}/metrics/{self._name_path}')

        df_feature_importance = pd.DataFrame({'variable': grid_search.feature_names_in_, 'importance': grid_search.best_estimator_.feature_importances_})
        df_feature_importance.to_csv(f'{self._output_path_train}/feature_importance/{self._name_path}/feature_importance.csv', index=False)
        mlflow.log_artifact(f'{self._output_path_train}/feature_importance/{self._name_path}/feature_importance.csv', artifact_path=f'{self._output_path_train}/feature_importance/{self._name_path}')

        with open(f'{self._output_path_train}/models/{self._name_path}/grid_search_model.pickle', 'wb') as handle:
            pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(f'{self._output_path_train}/models/{self._name_path}/grid_search_model.pickle', artifact_path=f'{self._output_path_train}/models/{self._name_path}')
    
def process_train_evaluate_models(model_parameters_grid, model, name_path, preprocess_train_run_id):
    if (os.getcwd().endswith("src")):
        os.chdir("..")
    clientes_train_path = download_artifacts(run_id=preprocess_train_run_id, artifact_path="data/out/data_train_prepared.csv")
    df_data_train = pd.read_csv(clientes_train_path)
    train_validate_models_instance = TrainEvaluateModels(output_path_train="outputs/train", output_path_preprocess="outputs/preprocess", name_path=name_path, preprocess_train_run_id=preprocess_train_run_id)
    train_validate_models_instance.train_evaluate_models(df_data_train, model_parameters_grid, model)

def train_models(tracking_uri, experiment_name, parent_run_id, preprocess_train_run_id):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="train_evaluate_models", nested=True) as train_evaluate_models:
            train_evaluate_models_id = train_evaluate_models.info.run_id
            model_parameters_grid_xgbost = {'max_depth': [3, 5, 7],'eta': [0.05, 0.1, 0.2], 'gamma': [0, 1, 5], 'min_child_weight': [1, 5, 10],
            'subsample': [0.6, 0.8, 1.0], 'n_estimators': [50, 100], 'scale_pos_weight': [1, 5, 10]}
            xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='auc',random_state=42)
            mlflow.log_param("xgbost_param_grid", str(model_parameters_grid_xgbost))
            model_parameters_grid_random_forest = {'n_estimators': [50, 100, 200], 'max_depth': [None, 4, 6, 8],              
            'min_samples_leaf': [1, 10, 50], 'min_impurity_decrease': [0.0, 0.01, 0.05]}
            rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            mlflow.log_param("random_forest_param_grid", str(model_parameters_grid_random_forest))
            process_train_evaluate_models(model_parameters_grid_xgbost,xgb_model,'xgbost',preprocess_train_run_id)
            process_train_evaluate_models(model_parameters_grid_random_forest,rf_model, 'random_forest',preprocess_train_run_id)
    return train_evaluate_models_id

