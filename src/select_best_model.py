import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, log_loss, ConfusionMatrixDisplay, roc_curve
import os
import pickle
import mlflow
from mlflow.artifacts import download_artifacts
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

class SelectBestModel:
    _output_path_train = ""
    _best_model = None
    _name_path = ""
    _train_evaluate_models_id = ""

    def __init__(self, output_path_train, name_path, train_evaluate_models_id):
        self._output_path_train = output_path_train
        self._name_path = name_path
        self._train_evaluate_models_id = train_evaluate_models_id

    def _get_features_name(self):
        feature_path = download_artifacts(run_id=self._train_evaluate_models_id, artifact_path=f'{self._output_path_train}/feature_importance/{self._name_path}/feature_importance.csv')
        df_feature_importance = pd.read_csv(feature_path)    
        return df_feature_importance['variable'].to_list()

    def _get_target_name(self):
        y_col = "ATTRITION"
        return y_col

    def _evaluate_best_model_in_dataset(self, df_data):
        x_cols = self._get_features_name()
        y_col = self._get_target_name()
        y_proba = self._best_model.predict_proba(df_data[x_cols])[:,1]
        
        fpr, tpr, thresholds = roc_curve(df_data[y_col], y_proba)
        j_scores = tpr - fpr
        optimal_idx = j_scores.argmax()
        best_threshold = thresholds[optimal_idx]
        y_pred_label = (y_proba >= best_threshold).astype(int)

        report = classification_report(df_data[y_col], y_pred_label, output_dict=True)
        auc_metric = roc_auc_score(df_data[y_col],y_proba)
        mlflow.log_metric(f"{self._name_path}_roc_auc_test", auc_metric)
        mlflow.log_metric(f"{self._name_path}_precision_test", report["1"]["precision"])
        mlflow.log_metric(f"{self._name_path}_recall_test", report["1"]["recall"])
        mlflow.log_metric(f"{self._name_path}_f1-score_test", report["1"]["f1-score"])
        mlflow.log_metric(f"{self._name_path}_log_loss_test", log_loss(df_data[y_col], y_proba))

        roc_display = RocCurveDisplay.from_predictions(df_data[y_col], y_proba)
        plt.title(f"Receiver Operating Characteristic (ROC) Curve for {self._name_path}")
        plt.savefig(f'{self._output_path_train}/metrics/{self._name_path}/roc_curve_{self._name_path}_test.png')
        plt.close()
        mlflow.log_artifact(f'{self._output_path_train}/metrics/{self._name_path}/roc_curve_{self._name_path}_test.png', artifact_path=f'{self._output_path_train}/metrics/{self._name_path}')
        
        disp = ConfusionMatrixDisplay.from_predictions(df_data[y_col], y_pred_label)
        plt.title(f"Confusion Matrix for {self._name_path}")
        plt.savefig(f'{self._output_path_train}/metrics/{self._name_path}/conf_matrix_{self._name_path}_test.png')
        plt.close()
        mlflow.log_artifact(f'{self._output_path_train}/metrics/{self._name_path}/conf_matrix_{self._name_path}_test.png', artifact_path=f'{self._output_path_train}/metrics/{self._name_path}')

        return report["1"]["f1-score"]

    def select_best_model(self, df_data_train, df_data_test):
        path_grid = download_artifacts(run_id=self._train_evaluate_models_id, artifact_path=f'{self._output_path_train}/models/{self._name_path}/grid_search_model.pickle')
        with open(path_grid, 'rb') as handle:
            grid_search = pickle.load(handle)

        best_params = grid_search.best_params_
        if self._name_path == "xgbost":
            model = XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42)
        elif self._name_path == "random_forest":
            model = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
        
        x_cols = self._get_features_name()
        y_col = self._get_target_name()
        model.fit(df_data_train[x_cols], df_data_train[y_col].values.ravel())
        self._best_model = model
        signature = infer_signature(df_data_train[x_cols], self._best_model.predict_proba(df_data_train[x_cols]))
        input_example = df_data_train[x_cols].iloc[:5]
        if self._name_path == "xgbost":
            mlflow.xgboost.log_model(self._best_model, artifact_path=f"{self._name_path}_model", signature=signature,input_example=input_example)
        elif self._name_path == "random_forest":
            mlflow.sklearn.log_model(self._best_model, artifact_path=f"{self._name_path}_model", signature=signature,input_example=input_example)

        with open(f'{self._output_path_train}/models/{self._name_path}/best_model.pickle', 'wb') as handle:
            pickle.dump(self._best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(f'{self._output_path_train}/models/{self._name_path}/best_model.pickle', artifact_path=f'{self._output_path_train}/models/{self._name_path}')

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{self._name_path}_model"
        model_registry_name = f"{self._name_path}_registered_model"
        result = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
        client = MlflowClient()
        
        client.set_model_version_tag(
        name=model_registry_name,
        version=result.version,
        key="estado",
        value="development")

        client.set_registered_model_alias(
        name=model_registry_name,
        alias="challenger",
        version=result.version)

        client.update_model_version(
        name=model_registry_name,
        version=result.version,
        description=f"{self._name_path} entrenado sobre todo el conjunto de train con los mejores hiperparametros segun el auc.")
        
        f1_score_metric_test = self._evaluate_best_model_in_dataset(df_data_test)

        df_metrics = pd.DataFrame({'sample':['test'],'f1_score':[f1_score_metric_test]})
        df_metrics.to_csv(f'{self._output_path_train}/metrics/{self._name_path}/test_metrics.csv', index=False)
        mlflow.log_artifact(f'{self._output_path_train}/metrics/{self._name_path}/test_metrics.csv', artifact_path=f'{self._output_path_train}/metrics/{self._name_path}')

def process_select_best_model(tracking_uri, experiment_name, parent_run_id, preprocess_train_run_id, prepare_dataset_test_id, train_evaluate_models_id):
    if (os.getcwd().endswith("src")):
        os.chdir("..")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="select_best_model", nested=True) as select_best_model:
            select_best_model_id = select_best_model.info.run_id
            clientes_train_path = download_artifacts(run_id=preprocess_train_run_id, artifact_path="data/out/data_train_prepared.csv")
            df_data_train = pd.read_csv(clientes_train_path)
            clientes_test_path = download_artifacts(run_id=prepare_dataset_test_id, artifact_path="data/out/data_test_prepared.csv")
            df_data_test = pd.read_csv(clientes_test_path)
            select_best_model_instance = SelectBestModel(output_path_train="outputs/train", name_path='xgbost',train_evaluate_models_id=train_evaluate_models_id)
            select_best_model_instance.select_best_model(df_data_train, df_data_test)
            select_best_model_instance = SelectBestModel(output_path_train="outputs/train", name_path='random_forest',train_evaluate_models_id=train_evaluate_models_id)
            select_best_model_instance.select_best_model(df_data_train, df_data_test)
    return select_best_model_id
