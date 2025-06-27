import fire
import pandas as pd
import sklearn.metrics as metrics
import os
import pickle

class SelectBestModel:
    _output_path_train = ""
    _best_model = None
    _name_path = ""

    def __init__(self, output_path_train, name_path):
        self._output_path_train = output_path_train
        self._name_path = name_path

    def _get_features_name(self):
        df_feature_importance = pd.read_csv(f'{self._output_path_train}/feature_importance/{self._name_path}/feature_importance.csv')
        return df_feature_importance['variable'].to_list()

    def _get_target_name(self):
        y_col = "ATTRITION"
        return y_col

    def _evaluate_best_model_in_dataset(self, df_data):
        x_cols = self._get_features_name()
        y_col = self._get_target_name()
        y_pred = self._best_model.predict_proba(df_data[x_cols])
        auc_metric = metrics.roc_auc_score(df_data[y_col], y_pred[:,1])
        return auc_metric

    def select_best_model(self, df_data_train, df_data_test):
        with open(f'{self._output_path_train}/models/{self._name_path}/grid_search_model.pickle', 'rb') as handle:
            grid_search = pickle.load(handle)

        self._best_model = grid_search.best_estimator_
        with open(f'{self._output_path_train}/models/{self._name_path}/best_model.pickle', 'wb') as handle:
            pickle.dump(self._best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        auc_metric_train = self._evaluate_best_model_in_dataset(df_data_train)
        auc_metric_test = self._evaluate_best_model_in_dataset(df_data_test)

        df_metrics = pd.DataFrame({'sample':['train','test'],'auc':[auc_metric_train, auc_metric_test]})
        df_metrics.to_csv(f'{self._output_path_train}/metrics/{self._name_path}/train_test_metrics.csv', index=False)


def process_select_best_model():
    if (os.getcwd().endswith("src")):
        os.chdir("..")
    df_data_train = pd.read_csv("data/out/data_train_prepared.csv")
    df_data_test = pd.read_csv("data/out/data_test_prepared.csv")
    select_best_model_instance = SelectBestModel(output_path_train="outputs/train", name_path='xgbost')
    select_best_model_instance.select_best_model(df_data_train, df_data_test)
    select_best_model_instance = SelectBestModel(output_path_train="outputs/train", name_path='random_forest')
    select_best_model_instance.select_best_model(df_data_train, df_data_test)

def main():
    process_select_best_model()

if __name__ == "__main__":
    fire.Fire(main)
