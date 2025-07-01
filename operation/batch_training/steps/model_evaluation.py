from batch_training_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step

instance_type = "ml.m5.2xlarge"
default_path = DEFAULT_PATH
image_uri = "762233743642.dkr.ecr.us-east-2.amazonaws.com/vecolazarte/batch-training@sha256:8301d35f2fff625f01304ef65977b41a074860fa441b1ac19a266595fa9bac27"


@step(
    name="ModelEvaluation",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def evaluate(experiment_name: str, run_id: str, data_pull_id: str, training_run_id: str) -> tuple[str, str, str]:
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'awswrangler==3.12.0', 'sagemaker==2.244.0']) 
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
    
    # Funciones Best model

    def get_features_name(name_path,training_run_id):
        feature_path = download_artifacts(run_id=training_run_id, artifact_path=f'outputs/train/feature_importance/{name_path}/feature_importance.csv')
        df_feature_importance = pd.read_csv(feature_path)    
        return df_feature_importance['variable'].to_list()

    def get_target_name():
        y_col = "ATTRITION"
        return y_col

    def evaluate_best_model_in_dataset(df_data,name_path,best_model,training_run_id):
        x_cols = get_features_name(name_path,training_run_id)
        y_col = get_target_name()
        y_proba = best_model.predict_proba(df_data[x_cols])[:,1]
        
        fpr, tpr, thresholds = roc_curve(df_data[y_col], y_proba)
        j_scores = tpr - fpr
        optimal_idx = j_scores.argmax()
        best_threshold = thresholds[optimal_idx]
        y_pred_label = (y_proba >= best_threshold).astype(int)

        report = classification_report(df_data[y_col], y_pred_label, output_dict=True)
        auc_metric = roc_auc_score(df_data[y_col],y_proba)
        mlflow.log_metric(f"{name_path}_roc_auc_test", auc_metric)
        mlflow.log_metric(f"{name_path}_precision_test", report["1"]["precision"])
        mlflow.log_metric(f"{name_path}_recall_test", report["1"]["recall"])
        mlflow.log_metric(f"{name_path}_f1-score_test", report["1"]["f1-score"])
        mlflow.log_metric(f"{name_path}_log_loss_test", log_loss(df_data[y_col], y_proba))

        roc_display = RocCurveDisplay.from_predictions(df_data[y_col], y_proba)
        plt.title(f"Receiver Operating Characteristic (ROC) Curve for {name_path}")
        plt.savefig(os.path.join(output_dir,f'roc_curve_{name_path}_test.png'))
        plt.close()
        mlflow.log_artifact(os.path.join(output_dir, f"roc_curve_{name_path}_test.png"), artifact_path=f"outputs/train/metrics/{name_path}")
        wr.s3.upload(local_file=os.path.join(output_dir, f"roc_curve_{name_path}_test.png"),path=os.path.join(train_s3_path, "outputs", "train", "metrics", name_path, f"roc_curve_{name_path}_test.png"))
        
        disp = ConfusionMatrixDisplay.from_predictions(df_data[y_col], y_pred_label)
        plt.title(f"Confusion Matrix for {name_path}")
        plt.savefig(os.path.join(output_dir,f'conf_matrix_{name_path}_test.png'))
        plt.close()
        mlflow.log_artifact(os.path.join(output_dir, f"conf_matrix_{name_path}_test.png"), artifact_path=f"outputs/train/metrics/{name_path}")
        wr.s3.upload(local_file=os.path.join(output_dir, f"conf_matrix_{name_path}_test.png"),path=os.path.join(train_s3_path, "outputs", "train", "metrics", name_path, f"conf_matrix_{name_path}_test.png"))
        
        return report["1"]["recall"]

    def select_best_model(df_data_train, df_data_test,name_path,training_run_id):
        path_grid = download_artifacts(run_id=training_run_id, artifact_path=f'outputs/train/models/{name_path}/grid_search_model.pickle')
        with open(path_grid, 'rb') as handle:
            grid_search = pickle.load(handle)

        best_params = grid_search.best_params_
        if name_path == "xgbost":
            model = XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42)
        elif name_path == "random_forest":
            model = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
        
        x_cols = get_features_name(name_path,training_run_id)
        y_col = get_target_name()
        model.fit(df_data_train[x_cols], df_data_train[y_col].values.ravel())
        best_model = model
        signature = infer_signature(df_data_train[x_cols], best_model.predict_proba(df_data_train[x_cols]))
        input_example = df_data_train[x_cols].iloc[:5]
        if name_path == "xgbost":
            mlflow.xgboost.log_model(best_model, artifact_path=f"{name_path}_model", signature=signature,input_example=input_example)
        elif name_path == "random_forest":
            mlflow.sklearn.log_model(best_model, artifact_path=f"{name_path}_model", signature=signature,input_example=input_example)

        with open(os.path.join(output_dir, "best_model.pickle"), 'wb') as handle:
            pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(os.path.join(output_dir, "best_model.pickle"), artifact_path=f"outputs/train/models/{name_path}")
        wr.s3.upload(local_file=os.path.join(output_dir, "best_model.pickle"),path=os.path.join(train_s3_path, "outputs", "train", "models", name_path, "best_model.pickle"))
        
        recall_metric_test = evaluate_best_model_in_dataset(df_data_test,name_path,best_model,training_run_id)
        df_metrics = pd.DataFrame({'sample':['test'],'recall':[recall_metric_test]})

        df_metrics.to_csv(os.path.join(train_s3_path, "outputs", "train", "metrics", name_path, "test_metrics.csv"), index=False)
        df_metrics.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "test_metrics.csv"), artifact_path=f"outputs/train/metrics/{name_path}")

        return recall_metric_test

    
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelEvaluation", nested=True) as evaluation_run:
            evaluation_run_id = evaluation_run.info.run_id
            
            clientes_train_path = download_artifacts(run_id=data_pull_id, artifact_path="data/out/data_train_prepared.csv")
            df_data_train = pd.read_csv(clientes_train_path)

            path_train_clientes = os.path.join(train_s3_path, "data", "out", "data_train_prepared.csv")
            mlflow.log_input(mlflow.data.from_pandas(df_data_train, path_train_clientes, targets='ATTRITION'), context="data_train")
            
            clientes_test_path = download_artifacts(run_id=data_pull_id, artifact_path="data/out/data_test_prepared.csv")
            df_data_test = pd.read_csv(clientes_test_path)

            path_test_clientes = os.path.join(train_s3_path, "data", "out", "data_test_prepared.csv")
            mlflow.log_input(mlflow.data.from_pandas(df_data_test, path_test_clientes, targets='ATTRITION'), context="data_test")
            
            metrics_rf = select_best_model(df_data_train, df_data_test, 'random_forest',training_run_id)
            metrics_xg = select_best_model(df_data_train, df_data_test, 'xgbost',training_run_id)
    return evaluation_run_id, metrics_rf, metrics_xg