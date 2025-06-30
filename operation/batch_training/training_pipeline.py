from sagemaker.workflow.function_step import step
from sagemaker.workflow.pipeline import Pipeline
import sagemaker
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep

from steps.data_pull import data_pull
from steps.model_evaluation import evaluate
from steps.model_registration_rf import register_random_forest_model
from steps.model_registration_xgb import register_xgboost_model
from steps.model_training import model_training

from batch_training_utils import MODEL_NAME, USERNAME, ENV_CODE, PIPELINE_NAME, SAGEMAKER_ROLE

#MLFlow setting
experiment_name = f"pipeline-train-{ENV_CODE}-{USERNAME}"

# Parameter setting
cod_month = ParameterString(name="PeriodoCargaClientes")
cod_month_start = ParameterInteger(name="PeriodoCargaRequerimientosInicio")
cod_month_end = ParameterInteger(name="PeriodoCargaRequerimientosFin")

# Steps setting
data_pull_step = data_pull(
    experiment_name=experiment_name,
    run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    cod_month=cod_month,
    cod_month_start=cod_month_start,
    cod_month_end=cod_month_end
)

model_training_step = model_training(
    experiment_name=experiment_name,
    run_id=data_pull_step[0],
    data_pull_id=data_pull_step[1]
)

model_evaluation_step = evaluate(
    experiment_name=experiment_name,
    run_id=data_pull_step[0],
    data_pull_id=data_pull_step[1],
    training_run_id=model_training_step
)

conditional_register_step = ConditionStep(
    name="ConditionalRegisterOverall",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=model_evaluation_step[1], 
            right=model_evaluation_step[2], 
        )
    ],
    if_steps=[
        ConditionStep(
            name="ConditionalRegisterRandomForestBranch",
            conditions=[
                ConditionGreaterThanOrEqualTo(
                    left=model_evaluation_step[1], 
                    right=0.6,
                )
            ],
            if_steps=[
                register_random_forest_model( 
                    experiment_name=experiment_name,
                    name_path='random_forest',
                    run_id=data_pull_step[0],
                    evaluation_run_id = model_evaluation_step[0]
                )
            ],
            else_steps=[
                FailStep(
                    name="FailRandomForestPerformance", 
                    error_message="Random Forest performance is not good enough"
                )
            ]
        )
    ],
    else_steps=[
        ConditionStep(
            name="ConditionalRegisterXGBoostBranch",
            conditions=[
                ConditionGreaterThanOrEqualTo(
                    left=model_evaluation_step[2], 
                    right=0.6,
                )
            ],
            if_steps=[
                register_xgboost_model(
                    experiment_name=experiment_name,
                    name_path='xgbost', 
                    run_id=data_pull_step[0],
                    evaluation_run_id = model_evaluation_step[0]
                )
            ],
            else_steps=[
                FailStep(
                    name="FailXGBoostPerformance", 
                    error_message="XGBoost performance is not good enough"
                )
            ]
        )
    ]
)


# Pipeline creation
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[
        data_pull_step,
        model_training_step,
        model_evaluation_step,
        conditional_register_step
    ],
    parameters=[cod_month, cod_month_start, cod_month_end]
)

pipeline.upsert(role_arn=SAGEMAKER_ROLE)
