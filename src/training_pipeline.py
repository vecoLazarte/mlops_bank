from split_dataset import run_split_dataset
from preprocess_dataset_train import process_preprocess_dataset
from prepare_dataset_test import process_prepare_dataset
from train_evaluate_models import train_models
from select_best_model import process_select_best_model
from score_model import process_score_model

experiment_name = f"pipeline-train-attrition-detection_1"
run_name = '255'
tracking_uri = "http://localhost:5000"

parent_run_id, split_run_id = run_split_dataset(tracking_uri = tracking_uri, experiment_name=experiment_name, run_name=run_name)
preprocess_train_run_id = process_preprocess_dataset(tracking_uri = tracking_uri, experiment_name=experiment_name, parent_run_id=parent_run_id, split_run_id=split_run_id)
prepare_dataset_test_id = process_prepare_dataset(tracking_uri = tracking_uri, experiment_name=experiment_name, parent_run_id=parent_run_id, split_run_id=split_run_id, preprocess_train_run_id=preprocess_train_run_id)
train_evaluate_models_id = train_models(tracking_uri = tracking_uri, experiment_name=experiment_name, parent_run_id=parent_run_id, preprocess_train_run_id=preprocess_train_run_id)
select_best_model_id = process_select_best_model(tracking_uri = tracking_uri, experiment_name=experiment_name, parent_run_id=parent_run_id, preprocess_train_run_id=preprocess_train_run_id, prepare_dataset_test_id=prepare_dataset_test_id, train_evaluate_models_id=train_evaluate_models_id)
score_model_id = process_score_model(tracking_uri = tracking_uri, experiment_name=experiment_name, parent_run_id=parent_run_id, preprocess_train_run_id=preprocess_train_run_id, select_best_model_id=select_best_model_id, train_evaluate_models_id=train_evaluate_models_id)