from batch_inference_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE, default_prefix, DEFAULT_BUCKET
from sagemaker.workflow.function_step import step

# Global variables
instance_type = "ml.m5.2xlarge"
default_bucket = DEFAULT_BUCKET
# Step definition
@step(
    name="DataPull",
    instance_type=instance_type,
    role=SAGEMAKER_ROLE
)
def data_pull(experiment_name: str, run_name: str, cod_month: str, cod_month_start: str, cod_month_end: str) -> str:
    import mlflow
    from mlflow.artifacts import download_artifacts
    import subprocess
    subprocess.run(['pip', 'install', 'awswrangler==3.12.0']) 
    import awswrangler as wr
    import os
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    import tempfile
    output_dir = tempfile.mkdtemp()
    import boto3
    s3 = boto3.client('s3')

    TARGET_COL = 'ATTRITION'
    query_clientes = """
       SELECT
        TRY_CAST(id_correlativo AS BIGINT) AS id_correlativo,
        TRY_CAST(codmes AS BIGINT) AS codmes,
        TRY_CAST(flg_bancarizado AS BIGINT) AS flg_bancarizado,
        rang_ingreso,
        flag_lima_provincia,
        TRY_CAST(edad AS DOUBLE) AS edad,
        TRY_CAST(antiguedad AS DOUBLE) AS antiguedad,
        rang_sdo_pasivo_menos0,
        TRY_CAST(sdo_activo_menos0 AS BIGINT) AS sdo_activo_menos0,
        TRY_CAST(sdo_activo_menos1 AS BIGINT) AS sdo_activo_menos1,
        TRY_CAST(sdo_activo_menos2 AS BIGINT) AS sdo_activo_menos2,
        TRY_CAST(sdo_activo_menos3 AS BIGINT) AS sdo_activo_menos3,
        TRY_CAST(sdo_activo_menos4 AS BIGINT) AS sdo_activo_menos4,
        TRY_CAST(sdo_activo_menos5 AS BIGINT) AS sdo_activo_menos5,
        TRY_CAST(flg_seguro_menos0 AS BIGINT) AS flg_seguro_menos0,
        TRY_CAST(flg_seguro_menos1 AS BIGINT) AS flg_seguro_menos1,
        TRY_CAST(flg_seguro_menos2 AS BIGINT) AS flg_seguro_menos2,
        TRY_CAST(flg_seguro_menos3 AS BIGINT) AS flg_seguro_menos3,
        TRY_CAST(flg_seguro_menos4 AS BIGINT) AS flg_seguro_menos4,
        TRY_CAST(flg_seguro_menos5 AS BIGINT) AS flg_seguro_menos5,
        rang_nro_productos_menos0,
        TRY_CAST(flg_nomina AS BIGINT) AS flg_nomina,
        TRY_CAST(nro_acces_canal1_menos0 AS BIGINT) AS nro_acces_canal1_menos0,
        TRY_CAST(nro_acces_canal1_menos1 AS BIGINT) AS nro_acces_canal1_menos1,
        TRY_CAST(nro_acces_canal1_menos2 AS BIGINT) AS nro_acces_canal1_menos2,
        TRY_CAST(nro_acces_canal1_menos3 AS BIGINT) AS nro_acces_canal1_menos3,
        TRY_CAST(nro_acces_canal1_menos4 AS BIGINT) AS nro_acces_canal1_menos4,
        TRY_CAST(nro_acces_canal1_menos5 AS BIGINT) AS nro_acces_canal1_menos5,
        TRY_CAST(nro_acces_canal2_menos0 AS BIGINT) AS nro_acces_canal2_menos0,
        TRY_CAST(nro_acces_canal2_menos1 AS BIGINT) AS nro_acces_canal2_menos1,
        TRY_CAST(nro_acces_canal2_menos2 AS BIGINT) AS nro_acces_canal2_menos2,
        TRY_CAST(nro_acces_canal2_menos3 AS BIGINT) AS nro_acces_canal2_menos3,
        TRY_CAST(nro_acces_canal2_menos4 AS BIGINT) AS nro_acces_canal2_menos4,
        TRY_CAST(nro_acces_canal2_menos5 AS BIGINT) AS nro_acces_canal2_menos5,
        TRY_CAST(nro_acces_canal3_menos0 AS BIGINT) AS nro_acces_canal3_menos0,
        TRY_CAST(nro_acces_canal3_menos1 AS BIGINT) AS nro_acces_canal3_menos1,
        TRY_CAST(nro_acces_canal3_menos2 AS BIGINT) AS nro_acces_canal3_menos2,
        TRY_CAST(nro_acces_canal3_menos3 AS BIGINT) AS nro_acces_canal3_menos3,
        TRY_CAST(nro_acces_canal3_menos4 AS BIGINT) AS nro_acces_canal3_menos4,
        TRY_CAST(nro_acces_canal3_menos5 AS BIGINT) AS nro_acces_canal3_menos5,
        TRY_CAST(nro_entid_ssff_menos0 AS BIGINT) AS nro_entid_ssff_menos0,
        TRY_CAST(nro_entid_ssff_menos1 AS BIGINT) AS nro_entid_ssff_menos1,
        TRY_CAST(nro_entid_ssff_menos2 AS BIGINT) AS nro_entid_ssff_menos2,
        TRY_CAST(nro_entid_ssff_menos3 AS BIGINT) AS nro_entid_ssff_menos3,
        TRY_CAST(nro_entid_ssff_menos4 AS BIGINT) AS nro_entid_ssff_menos4,
        TRY_CAST(nro_entid_ssff_menos5 AS BIGINT) AS nro_entid_ssff_menos5,
        TRY_CAST(flg_sdo_otssff_menos0 AS BIGINT) AS flg_sdo_otssff_menos0,
        TRY_CAST(flg_sdo_otssff_menos1 AS BIGINT) AS flg_sdo_otssff_menos1,
        TRY_CAST(flg_sdo_otssff_menos2 AS BIGINT) AS flg_sdo_otssff_menos2,
        TRY_CAST(flg_sdo_otssff_menos3 AS BIGINT) AS flg_sdo_otssff_menos3,
        TRY_CAST(flg_sdo_otssff_menos4 AS BIGINT) AS flg_sdo_otssff_menos4,
        TRY_CAST(flg_sdo_otssff_menos5 AS BIGINT) AS flg_sdo_otssff_menos5
        FROM oot_clientes_sample
        WHERE codmes = '{}';
    """.format(cod_month)
    
    query_requerimientos = """
        SELECT *
        FROM oot_requerimientos_sample
        WHERE codmes between {} and {};
        """.format(cod_month_start, cod_month_end)
    
    train_s3_path = f"s3://{DEFAULT_PATH}"

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    def prepare_impute_missing(df_data, x_cols):
        df_data_imputed = df_data.copy()
        s3_key = f'{default_prefix}/outputs/preprocess/imputacion_parametros.csv'
        local_path = 'imputacion_parametros.csv'
        s3.download_file(default_bucket, s3_key, local_path)
        df_impute_parameters = pd.read_csv(local_path)
        for col in x_cols:
            impute_value = df_impute_parameters[df_impute_parameters["variable"]==col]["valor"].values[0]
            df_data_imputed[col] = df_data_imputed[col].fillna(impute_value)
        return df_data_imputed
       
    def generar_variables_ingenieria(clientes_df):
        clientes_df["VAR_SDO_ACTIVO_6M"] = clientes_df["SDO_ACTIVO_MENOS0"] - clientes_df["SDO_ACTIVO_MENOS5"]
        clientes_df["PROM_SDO_ACTIVO_0M_2M"] = clientes_df[[f"SDO_ACTIVO_MENOS{i}" for i in range(3)]].mean(axis=1)
        clientes_df["PROM_SDO_ACTIVO_3M_5M"] = clientes_df[[f"SDO_ACTIVO_MENOS{i}" for i in range(3, 6)]].mean(axis=1)
        clientes_df["VAR_SDO_ACTIVO_3M"] = clientes_df["PROM_SDO_ACTIVO_0M_2M"] - clientes_df["PROM_SDO_ACTIVO_3M_5M"]
        clientes_df["PROM_SDO_ACTIVO_6M"] = clientes_df[[f"SDO_ACTIVO_MENOS{i}" for i in range(6)]].mean(axis=1)
        clientes_df["MESES_CON_SEGURO"] = clientes_df[[f"FLG_SEGURO_MENOS{i}" for i in range(6)]].sum(axis=1)
        for canal in [1, 2, 3]:
            base = f"NRO_ACCES_CANAL{canal}_MENOS"
            clientes_df[f"VAR_NRO_ACCES_CANAL{canal}_6M"] = clientes_df[f"{base}0"] - clientes_df[f"{base}5"]
            clientes_df[f"PROM_NRO_ACCES_CANAL{canal}_6M"] = clientes_df[[f"{base}{i}" for i in range(6)]].mean(axis=1)
            clientes_df[f"PROM_NRO_ACCES_CANAL{canal}_0M_2M"] = clientes_df[[f"{base}{i}" for i in range(3)]].mean(axis=1)
            clientes_df[f"PROM_NRO_ACCES_CANAL{canal}_3M_5M"] = clientes_df[[f"{base}{i}" for i in range(3, 6)]].mean(axis=1)
            clientes_df[f"VAR_NRO_ACCES_CANAL{canal}_3M"] = (clientes_df[f"PROM_NRO_ACCES_CANAL{canal}_0M_2M"] - clientes_df[f"PROM_NRO_ACCES_CANAL{canal}_3M_5M"])
        clientes_df["PROM_NRO_ENTID_SSFF_6M"] = clientes_df[[f"NRO_ENTID_SSFF_MENOS{i}" for i in range(6)]].mean(axis=1)
        clientes_df["VAR_NRO_ENTID_SSFF_6M"] = clientes_df["NRO_ENTID_SSFF_MENOS0"] - clientes_df["NRO_ENTID_SSFF_MENOS5"]
        clientes_df["PROM_NRO_ENTID_SSFF_0M_2M"] = clientes_df[[f"NRO_ENTID_SSFF_MENOS{i}" for i in range(3)]].mean(axis=1)
        clientes_df["PROM_NRO_ENTID_SSFF_3M_5M"] = clientes_df[[f"NRO_ENTID_SSFF_MENOS{i}" for i in range(3, 6)]].mean(axis=1)
        clientes_df["VAR_NRO_ENTID_SSFF_3M"] = (clientes_df["PROM_NRO_ENTID_SSFF_0M_2M"] - clientes_df["PROM_NRO_ENTID_SSFF_3M_5M"])
        clientes_df["MESES_CON_SALDO"] = clientes_df[[f"FLG_SDO_OTSSFF_MENOS{i}" for i in range(6)]].sum(axis=1)
        return clientes_df

    def construir_variables_requerimientos(df_reqs, id_col='ID_CORRELATIVO'):
        total_reqs = df_reqs.groupby(id_col).size().rename('total_requerimientos')
        if not isinstance(total_reqs, pd.DataFrame):
            total_reqs = total_reqs.to_frame()
        n_tipo_req = df_reqs.groupby(id_col)['TIPO_REQUERIMIENTO2'].nunique().rename('nro_tipos_requerimiento').to_frame()
        n_dictamen = df_reqs.groupby(id_col)['DICTAMEN'].nunique().rename('nro_dictamenes').to_frame()
        n_producto = df_reqs.groupby(id_col)['PRODUCTO_SERVICIO_2'].nunique().rename('nro_productos_servicios').to_frame()
        n_submotivo = df_reqs.groupby(id_col)['SUBMOTIVO_2'].nunique().rename('nro_submotivos').to_frame()
        tipo_ohe = pd.get_dummies(df_reqs['TIPO_REQUERIMIENTO2'], prefix='tipo')
        tipo_ohe[id_col] = df_reqs[id_col]
        tipo_ohe = tipo_ohe.groupby(id_col).sum()
        dictamen_ohe = pd.get_dummies(df_reqs['DICTAMEN'], prefix='dictamen')
        dictamen_ohe[id_col] = df_reqs[id_col]
        dictamen_ohe = dictamen_ohe.groupby(id_col).sum()
        df_agregado = pd.concat([total_reqs, n_tipo_req, n_dictamen, n_producto, n_submotivo, tipo_ohe, dictamen_ohe],axis=1)
        return df_agregado
    
    def apply_label_encoders_to_test(df_test):
        df_test['RANG_SDO_PASIVO_MENOS0'] = df_test['RANG_SDO_PASIVO_MENOS0'].replace('Cero', 'Rango_SDO_00')
        df_test['FLAG_LIMA_PROVINCIA'] = df_test['FLAG_LIMA_PROVINCIA'].map({'Lima': 1, 'Provincia': 0})
        s3_key = f'{default_prefix}/outputs/preprocess/label_encoder_train.pkl'
        local_path = 'label_encoder_train.pkl'
        s3.download_file(default_bucket, s3_key, local_path)
        with open(local_path, 'rb') as f:
            encoders_clientes = pickle.load(f)
        for col, le in encoders_clientes.items():
            df_test[col] = le.transform(df_test[col])
        return df_test
    
    def aplicar_estandarizacion_test(df_test):
        s3_key = f'{default_prefix}/outputs/preprocess/scaler_train.pkl'
        local_path = 'scaler_train.pkl'
        s3.download_file(default_bucket, s3_key, local_path)
        with open(local_path, 'rb') as f:
            scaler = pickle.load(f)
        no_escalar = ['ID_CORRELATIVO', 'CODMES']
        columnas_a_escalar = df_test.columns.difference(no_escalar)
        df_predictoras = df_test[columnas_a_escalar]
        df_escaladas = pd.DataFrame(scaler.transform(df_predictoras),columns=columnas_a_escalar,index=df_test.index)
        df_test_estandarizado = pd.concat([df_test[no_escalar], df_escaladas], axis=1)
        return df_test_estandarizado
    
    def prepare_dataset(df_data_test, df_requerimientos_test):
        x_cols_clientes = ['RANG_INGRESO','FLAG_LIMA_PROVINCIA','EDAD','ANTIGUEDAD']
        x_cols_requerimientos = ['DICTAMEN']
        df_data_imputed_clientes = prepare_impute_missing(df_data_test, x_cols_clientes)
        df_data_imputed_requerimientos = prepare_impute_missing(df_requerimientos_test, x_cols_requerimientos)
        df_data_feature_clientes = generar_variables_ingenieria(df_data_imputed_clientes)
        df_data_feature_requerimientos = construir_variables_requerimientos(df_data_imputed_requerimientos)
        df_data_encoder_clientes = apply_label_encoders_to_test(df_data_feature_clientes)
        df_final = df_data_encoder_clientes.merge(df_data_feature_requerimientos, on='ID_CORRELATIVO', how='left')
        df_final.fillna(0, inplace=True)
        df_final = aplicar_estandarizacion_test(df_final)
        return df_final

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="DataPull", nested=True) as data_pull:
            data_pull_id = data_pull.info.run_id
            df_data_test = wr.athena.read_sql_query(sql=query_clientes, database="bank_attrition")
            df_data_test.columns = df_data_test.columns.str.upper()
            df_data_test['RANG_INGRESO'] = df_data_test['RANG_INGRESO'].replace('', np.nan)
            df_data_test['FLAG_LIMA_PROVINCIA'] = df_data_test['FLAG_LIMA_PROVINCIA'].replace('', np.nan) 

            df_requerimientos_test = wr.athena.read_sql_query(sql=query_requerimientos, database="bank_attrition")
            df_requerimientos_test.columns = df_requerimientos_test.columns.str.upper()
            df_requerimientos_test['DICTAMEN'] = df_requerimientos_test['DICTAMEN'].replace('', np.nan) 

            df_data_score_prepared = prepare_dataset(df_data_test, df_requerimientos_test)
            
            df_data_score_prepared.to_csv(os.path.join(train_s3_path, "inf-raw-data", f"df_data_score_prepared_{cod_month}.csv"), index=False)
            df_data_score_prepared.to_csv(os.path.join(output_dir, f"df_data_score_prepared_{cod_month}.csv"), index=False)
            mlflow.log_artifact(os.path.join(output_dir, f"df_data_score_prepared_{cod_month}.csv"), artifact_path=f"inf-raw-data")
            
    return run_id, data_pull_id