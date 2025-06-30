from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from steps.data_pull import data_pull
from steps.model_inference import model_inference
from steps.data_push import data_push


from batch_inference_utils import MODEL_NAME, USERNAME, ENV_CODE, PIPELINE_NAME, SAGEMAKER_ROLE

#MLFlow setting
experiment_name = f"pipeline-inference-{ENV_CODE}-{USERNAME}"

# Parameter setting
cod_month = ParameterString(name="PeriodoCargaClientes")
cod_month_start = ParameterInteger(name="PeriodoCargaRequerimientosInicio")
cod_month_end = ParameterInteger(name="PeriodoCargaRequerimientosFin")

# Steps setting
data_pull_step = data_pull(experiment_name=experiment_name,
                           run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
                           cod_month=cod_month,
                           cod_month_start=cod_month_start,
                           cod_month_end=cod_month_end)

model_inference_step = model_inference(experiment_name=experiment_name,
                                       run_id=data_pull_step[0],
                                       data_pull_id=data_pull_step[1],
                                       cod_month=cod_month)

data_push_step = data_push(experiment_name=experiment_name,
                            run_id=data_pull_step[0],
                            model_inference_id=model_inference_step,
                            cod_month=cod_month)


# Pipeline creation
pipeline = Pipeline(name=PIPELINE_NAME,
                    steps=[data_pull_step,model_inference_step,data_push_step],
                   parameters=[cod_month, cod_month_start, cod_month_end])
pipeline.upsert(role_arn=SAGEMAKER_ROLE)


