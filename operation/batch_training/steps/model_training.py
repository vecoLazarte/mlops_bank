from batch_training_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step

instance_type = "ml.m5.2xlarge"
default_path = DEFAULT_PATH
image_uri = "762233743642.dkr.ecr.us-east-2.amazonaws.com/vecolazarte/batch-training@sha256:9e072265d03dd6650f327ab86d37922f18d56c66417a3b2e84e80e93ca5eef74"


@step(
    name="ModelTraining",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def model_training(experiment_name: str, run_id: str, data_pull_id: str ) -> str:
    import subprocess
    subprocess.run(['pip', 'install', 'awswrangler==3.4.0'])
    import awswrangler as wr
    import mlflow
    from mlflow.artifacts import download_artifacts
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    import os
    import pickle
    import tempfile
    from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, log_loss, ConfusionMatrixDisplay, roc_curve
    import matplotlib.pyplot as plt
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.models.signature import infer_signature
    from mlflow.tracking import MlflowClient
    output_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)
    train_s3_path = f"s3://{default_path}"

    # Funciones train CV
    
    def get_preprocess_x_columns():
        x_cols_path = download_artifacts(run_id=data_pull_id, artifact_path='outputs/preprocess/x_col_names.csv')
        x_cols = pd.read_csv(x_cols_path)['x_col'].to_list()
        return x_cols

    def get_preprocess_y_column():
        y_col_path = download_artifacts(run_id=data_pull_id, artifact_path='outputs/preprocess/y_col_name.csv')
        y_col = pd.read_csv(y_col_path)['y_col'].to_list()
        return y_col

    def train_evaluate_models(df_data_train, model_parameters_grid, model, name_path):
        x_cols = get_preprocess_x_columns()
        y_col = get_preprocess_y_column()
        mlflow.log_param("set_train_cv_rows", df_data_train.shape[0])
        mlflow.log_param("set_train_cv_cols", x_cols)
        grid_search = GridSearchCV(estimator=model, param_grid=model_parameters_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(df_data_train[x_cols], df_data_train[y_col].values.ravel())
        prefixed_params = {f"best_param_{name_path}_{k}": v for k, v in grid_search.best_params_.items()}
        mlflow.log_params(prefixed_params)
        df_model_results = pd.DataFrame({'model_parameters': grid_search.cv_results_['params'],
                                         'model_rank': grid_search.cv_results_['rank_test_score'],
                                         'auc_score_mean': grid_search.cv_results_['mean_test_score'],
                                         'auc_score_std': grid_search.cv_results_['std_test_score']})
        df_model_results['auc_score_cv'] = df_model_results['auc_score_std'] / df_model_results['auc_score_mean']
        df_model_results.to_csv(os.path.join(train_s3_path, "outputs", "train", "metrics", name_path, "train_cv_model_results.csv"), index=False)
        df_model_results.to_csv(os.path.join(output_dir, "train_cv_model_results.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "train_cv_model_results.csv"), artifact_path=f"outputs/train/metrics/{name_path}")
        df_model_results_best_model = df_model_results[df_model_results['model_rank']==1].copy()
        best_auc_score_mean = df_model_results_best_model['auc_score_mean'].values[0]
        mlflow.log_metric(f"best_cv_train_{name_path}_auc_score_mean", best_auc_score_mean)
        best_auc_score_std = df_model_results_best_model['auc_score_std'].values[0]
        mlflow.log_metric(f"best_cv_train_{name_path}_auc_score_std", best_auc_score_std)
        best_auc_score_cv = df_model_results_best_model['auc_score_cv'].values[0]
        mlflow.log_metric(f"best_cv_train_{name_path}_auc_score_cv", best_auc_score_cv)
        df_model_results_best_model.to_csv(os.path.join(train_s3_path, "outputs", "train", "metrics", name_path, "train_cv_model_results_best_model.csv"), index=False)
        df_model_results_best_model.to_csv(os.path.join(output_dir, "train_cv_model_results_best_model.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "train_cv_model_results_best_model.csv"), artifact_path=f"outputs/train/metrics/{name_path}")
        df_feature_importance = pd.DataFrame({'variable': grid_search.feature_names_in_, 'importance': grid_search.best_estimator_.feature_importances_})
        df_feature_importance.to_csv(os.path.join(train_s3_path, "outputs", "train", "feature_importance", name_path, "feature_importance.csv"), index=False)
        df_feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "feature_importance.csv"), artifact_path=f"outputs/train/feature_importance/{name_path}")
        with open(os.path.join(output_dir, "grid_search_model.pickle"), 'wb') as handle:
            pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(os.path.join(output_dir, "grid_search_model.pickle"), artifact_path=f"outputs/train/models/{name_path}")
        wr.s3.upload(local_file=os.path.join(output_dir, "grid_search_model.pickle"),path=os.path.join(train_s3_path, "outputs", "train", "models", name_path, "grid_search_model.pickle"))
        
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelTraining", nested=True) as training_run:
            training_run_id = training_run.info.run_id
            
            model_parameters_grid_xgbost = {'max_depth': [3, 5],'eta': [0.05, 0.1],'gamma': [0, 1],'min_child_weight': [1, 5],'subsample': [0.8],'n_estimators': [50],'scale_pos_weight': [1, 5]}
            xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='auc',random_state=42)
            mlflow.log_param("xgbost_param_grid", str(model_parameters_grid_xgbost))
            model_parameters_grid_random_forest = {'n_estimators': [100],'max_depth': [None, 6],'min_samples_leaf': [1, 10],'min_impurity_decrease': [0.0, 0.01]}
            rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            mlflow.log_param("random_forest_param_grid", str(model_parameters_grid_random_forest))
            clientes_train_path = download_artifacts(run_id=data_pull_id, artifact_path="data/out/data_train_prepared.csv")
            df_data_train = pd.read_csv(clientes_train_path)
            path_train_clientes = os.path.join(train_s3_path, "data", "out", "data_train_prepared.csv")
            mlflow.log_input(mlflow.data.from_pandas(df_data_train, path_train_clientes, targets='ATTRITION'), context="ModelTraining_data_train_cv")
            train_evaluate_models(df_data_train, model_parameters_grid_xgbost, xgb_model, 'xgbost')
            train_evaluate_models(df_data_train, model_parameters_grid_random_forest, rf_model, 'random_forest')
            
    return training_run_id