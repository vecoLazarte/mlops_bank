from batch_training_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step

instance_type = "ml.m5.2xlarge"
default_path = DEFAULT_PATH
image_uri = "762233743642.dkr.ecr.us-east-2.amazonaws.com/vecolazarte/batch-training:latest@sha256:8301d35f2fff625f01304ef65977b41a074860fa441b1ac19a266595fa9bac27"
@step(
    name="DataPull",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def data_pull(experiment_name: str, run_name: str, cod_month: str, cod_month_start: int, cod_month_end: int) -> tuple[str, str]:
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'awswrangler==3.12.0', 'sagemaker==2.244.0'])
    import mlflow
    from mlflow.artifacts import download_artifacts
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)
 
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
        TRY_CAST(attrition AS BIGINT) AS attrition,
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
        FROM train_clientes_sample
        WHERE codmes = '{}';
    """.format(cod_month)

    query_requerimientos = """
        SELECT *
        FROM train_requerimientos
        WHERE codmes between {} and {};
        """.format(cod_month_start, cod_month_end)
    
    train_s3_path = f"s3://{default_path}"

    # Funciones extraccion de datos
    def buscar_indices_coincidentes(df_clientes):
        ids_comunes = list(set(df_clientes['ID_CORRELATIVO']))
        return ids_comunes

    def split_data(ids_comunes):
        return train_test_split(ids_comunes, test_size=0.3, random_state=42)

    # Funciones preprocesamiento
    def save_y_col_name(y_col):
        df_y_col_name = pd.DataFrame({'y_col':[y_col]})
        df_y_col_name.to_csv(os.path.join(train_s3_path, "outputs", "preprocess", "y_col_name.csv"), index=False)
        df_y_col_name.to_csv(os.path.join(output_dir, "y_col_name.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "y_col_name.csv"), artifact_path="outputs/preprocess")

    def save_x_col_names(df_final, y_col):
        x_cols = [col for col in df_final.columns if col != y_col and col not in ['ID_CORRELATIVO', 'CODMES']]
        df_x_col_names = pd.DataFrame({'x_col': x_cols})
        df_x_col_names.to_csv(os.path.join(train_s3_path, "outputs", "preprocess", "x_col_names.csv"), index=False)
        df_x_col_names.to_csv(os.path.join(output_dir, "x_col_names.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "x_col_names.csv"), artifact_path="outputs/preprocess")
        
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
    
    def imputacion_variables(clientes_df, requerimientos_df):
        imputaciones = []
        moda_rango = clientes_df['RANG_INGRESO'].mode()[0]
        clientes_df['RANG_INGRESO'] = clientes_df['RANG_INGRESO'].fillna(moda_rango)
        imputaciones.append({'dataframe': 'clientes', 'variable': 'RANG_INGRESO', 'estrategia': 'moda', 'valor': moda_rango})
        moda_lima = clientes_df['FLAG_LIMA_PROVINCIA'].mode()[0]
        clientes_df['FLAG_LIMA_PROVINCIA'] = clientes_df['FLAG_LIMA_PROVINCIA'].fillna(moda_lima)
        imputaciones.append({'dataframe': 'clientes', 'variable': 'FLAG_LIMA_PROVINCIA', 'estrategia': 'moda', 'valor': moda_lima})
        mediana_edad = clientes_df['EDAD'].median()
        clientes_df['EDAD'] = clientes_df['EDAD'].fillna(mediana_edad)
        imputaciones.append({'dataframe': 'clientes', 'variable': 'EDAD', 'estrategia': 'mediana', 'valor': mediana_edad})
        mediana_antig = clientes_df['ANTIGUEDAD'].median()
        clientes_df['ANTIGUEDAD'] = clientes_df['ANTIGUEDAD'].fillna(mediana_antig)
        imputaciones.append({'dataframe': 'clientes', 'variable': 'ANTIGUEDAD', 'estrategia': 'mediana', 'valor': mediana_antig})
        moda_dictamen = requerimientos_df['DICTAMEN'].mode()[0]
        requerimientos_df['DICTAMEN'] = requerimientos_df['DICTAMEN'].fillna(moda_dictamen)
        imputaciones.append({'dataframe': 'requerimientos', 'variable': 'DICTAMEN', 'estrategia': 'moda', 'valor': moda_dictamen})
        df_imputaciones = pd.DataFrame(imputaciones)
        df_imputaciones.to_csv(os.path.join(train_s3_path, "outputs", "preprocess", "imputacion_parametros.csv"), index=False)
        df_imputaciones.to_csv(os.path.join(output_dir, "imputacion_parametros.csv"), index=False)
        mlflow.log_artifact(os.path.join(output_dir, "imputacion_parametros.csv"), artifact_path="outputs/preprocess")
        return clientes_df, requerimientos_df, df_imputaciones

    def encoder_categoricos(clientes_df):
        clientes_df['RANG_SDO_PASIVO_MENOS0'] = clientes_df['RANG_SDO_PASIVO_MENOS0'].replace('Cero', 'Rango_SDO_00')
        clientes_df['FLAG_LIMA_PROVINCIA'] = clientes_df['FLAG_LIMA_PROVINCIA'].map({'Lima': 1, 'Provincia': 0})
        cat_cols = clientes_df.select_dtypes(include=['object', 'category','string']).columns
        encoders_clientes = {} 
        for col in cat_cols:
            le = LabelEncoder()
            clientes_df[col] = le.fit_transform(clientes_df[col])
            encoders_clientes[col] = le
        return clientes_df, encoders_clientes

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

    def estandarizacion(df_final):
        no_escalar = ['ID_CORRELATIVO', 'CODMES', 'ATTRITION']
        columnas_a_escalar = df_final.columns.difference(no_escalar)
        df_predictoras = df_final[columnas_a_escalar]
        scaler = StandardScaler()
        df_escaladas = pd.DataFrame(scaler.fit_transform(df_predictoras),columns=columnas_a_escalar,index=df_final.index)
        df_final_estandarizado = pd.concat([df_final[no_escalar], df_escaladas],axis=1)
        return df_final_estandarizado, scaler

    def preprocess_dataset(clientes_df, requerimientos_df, y_col):
        save_y_col_name(y_col)
        clientes_df = generar_variables_ingenieria(clientes_df)
        clientes_df,requerimientos_df,df_imputaciones = imputacion_variables(clientes_df,requerimientos_df)
        clientes_df, artifact_encoders_clientes = encoder_categoricos(clientes_df)
        requerimientos_df = construir_variables_requerimientos(requerimientos_df)
        df_final = clientes_df.merge(requerimientos_df, on='ID_CORRELATIVO', how='left')
        df_final.fillna(0, inplace=True)
        df_final, artifact_scaler = estandarizacion(df_final)
        save_x_col_names(df_final, y_col)
        return df_final, artifact_encoders_clientes, artifact_scaler, df_imputaciones

    # Funciones preprocesamiento test

    def prepare_impute_missing_test(df_data, x_cols, df_impute_parameters):
        df_data_imputed = df_data.copy()
        for col in x_cols:
            impute_value = df_impute_parameters[df_impute_parameters["variable"]==col]["valor"].values[0]
            df_data_imputed[col] = df_data_imputed[col].fillna(impute_value)
        return df_data_imputed

    def apply_label_encoders_to_test(df_test, encoders_clientes):
        df_test['RANG_SDO_PASIVO_MENOS0'] = df_test['RANG_SDO_PASIVO_MENOS0'].replace('Cero', 'Rango_SDO_00')
        df_test['FLAG_LIMA_PROVINCIA'] = df_test['FLAG_LIMA_PROVINCIA'].map({'Lima': 1, 'Provincia': 0})
        for col, le in encoders_clientes.items():
            df_test[col] = le.transform(df_test[col])
        return df_test

    def aplicar_estandarizacion_test(df_test, scaler):
        no_escalar = ['ID_CORRELATIVO', 'CODMES', 'ATTRITION']
        columnas_a_escalar = df_test.columns.difference(no_escalar)
        df_predictoras = df_test[columnas_a_escalar]
        df_escaladas = pd.DataFrame(scaler.transform(df_predictoras),columns=columnas_a_escalar,index=df_test.index)
        df_test_estandarizado = pd.concat([df_test[no_escalar], df_escaladas], axis=1)
        return df_test_estandarizado

    def prepare_dataset_test(df_data_test,df_requerimientos_test,df_impute_parameters,encoders_clientes,scaler):
        x_cols_clientes = ['RANG_INGRESO','FLAG_LIMA_PROVINCIA','EDAD','ANTIGUEDAD']
        x_cols_requerimientos = ['DICTAMEN']
        df_data_imputed_clientes = prepare_impute_missing_test(df_data_test, x_cols_clientes, df_impute_parameters)
        df_data_imputed_requerimientos = prepare_impute_missing_test(df_requerimientos_test, x_cols_requerimientos, df_impute_parameters)
        df_data_feature_clientes = generar_variables_ingenieria(df_data_imputed_clientes)
        df_data_feature_requerimientos = construir_variables_requerimientos(df_data_imputed_requerimientos)
        df_data_encoder_clientes = apply_label_encoders_to_test(df_data_feature_clientes, encoders_clientes)
        df_final = df_data_encoder_clientes.merge(df_data_feature_requerimientos, on='ID_CORRELATIVO', how='left')
        df_final.fillna(0, inplace=True)
        df_final = aplicar_estandarizacion_test(df_final, scaler)
        return df_final
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        with mlflow.start_run(run_name="DataPull", nested=True) as data_pull:
            data_pull_id = data_pull.info.run_id

            # Ejecutamos funciones de extraccion de datos

            df_clientes = wr.athena.read_sql_query(sql=query_clientes, database="bank_attrition")
            df_clientes.columns = df_clientes.columns.str.upper()
            df_clientes['RANG_INGRESO'] = df_clientes['RANG_INGRESO'].replace('', np.nan)
            df_clientes['FLAG_LIMA_PROVINCIA'] = df_clientes['FLAG_LIMA_PROVINCIA'].replace('', np.nan) 

            df_requerimientos = wr.athena.read_sql_query(sql=query_requerimientos, database="bank_attrition")
            df_requerimientos.columns = df_requerimientos.columns.str.upper()
            df_requerimientos['DICTAMEN'] = df_requerimientos['DICTAMEN'].replace('', np.nan) 
            
            ids_comunes = buscar_indices_coincidentes(df_clientes)
            ids_train, ids_test = split_data(ids_comunes)

            train_clientes = df_clientes[df_clientes['ID_CORRELATIVO'].isin(ids_train)].copy()
            test_clientes = df_clientes[df_clientes['ID_CORRELATIVO'].isin(ids_test)].copy()

            train_requerimientos = df_requerimientos[df_requerimientos['ID_CORRELATIVO'].isin(ids_train)].copy()
            test_requerimientos = df_requerimientos[df_requerimientos['ID_CORRELATIVO'].isin(ids_test)].copy()

            path_train_clientes = os.path.join(train_s3_path, "data", "out", "clientes_data_train.csv")
            path_test_clientes = os.path.join(train_s3_path, "data", "out", "clientes_data_test.csv")
            path_train_reqs = os.path.join(train_s3_path, "data", "out", "requerimientos_data_train.csv")
            path_test_reqs = os.path.join(train_s3_path, "data", "out", "requerimientos_data_test.csv")

            train_clientes.to_csv(path_train_clientes, index=False)
            test_clientes.to_csv(path_test_clientes, index=False)
            train_requerimientos.to_csv(path_train_reqs, index=False)
            test_requerimientos.to_csv(path_test_reqs, index=False)

            train_clientes.to_csv(os.path.join(output_dir, "clientes_data_train.csv"), index=False)
            test_clientes.to_csv(os.path.join(output_dir, "clientes_data_test.csv"), index=False)
            train_requerimientos.to_csv(os.path.join(output_dir, "requerimientos_data_train.csv"), index=False)
            test_requerimientos.to_csv(os.path.join(output_dir, "requerimientos_data_test.csv"), index=False)

            mlflow.log_artifact(os.path.join(output_dir, "clientes_data_train.csv"), artifact_path="data/out")
            mlflow.log_artifact(os.path.join(output_dir, "clientes_data_test.csv"), artifact_path="data/out")
            mlflow.log_artifact(os.path.join(output_dir, "requerimientos_data_train.csv"), artifact_path="data/out")
            mlflow.log_artifact(os.path.join(output_dir, "requerimientos_data_test.csv"), artifact_path="data/out")

            mlflow.log_input(mlflow.data.from_pandas(train_clientes, path_train_clientes, targets=TARGET_COL), context="DataPull_train_clientes")
            mlflow.log_input(mlflow.data.from_pandas(train_requerimientos, path_train_reqs), context="DataPull_train_requerimientos")

            mlflow.log_input(mlflow.data.from_pandas(test_clientes, path_test_clientes, targets=TARGET_COL), context="DataPull_test_clientes")
            mlflow.log_input(mlflow.data.from_pandas(test_requerimientos, path_test_reqs), context="DataPull_test_requerimientos")

            # Ejecutamos las funciones de preprocesamiento

            df_data_train_prepared, artifact_encoders_clientes, artifact_scaler, df_imputaciones = preprocess_dataset(train_clientes, train_requerimientos, TARGET_COL)
            df_data_train_prepared.to_csv(os.path.join(train_s3_path, "data", "out", "data_train_prepared.csv"), index=False)
            df_data_train_prepared.to_csv(os.path.join(output_dir, "data_train_prepared.csv"), index=False)
            mlflow.log_artifact(os.path.join(output_dir, "data_train_prepared.csv"), artifact_path="data/out")
            
            with open(os.path.join(output_dir, "scaler_train.pkl"), 'wb') as f:
                pickle.dump(artifact_scaler, f)
            mlflow.log_artifact(os.path.join(output_dir, "scaler_train.pkl"), artifact_path="outputs/preprocess")

            with open(os.path.join(output_dir, "label_encoder_train.pkl"), 'wb') as f:
                pickle.dump(artifact_encoders_clientes, f)
            mlflow.log_artifact(os.path.join(output_dir, "label_encoder_train.pkl"), artifact_path="outputs/preprocess")

            wr.s3.upload(local_file=os.path.join(output_dir, "scaler_train.pkl"),path=os.path.join(train_s3_path, "outputs", "preprocess", "scaler_train.pkl"))
            wr.s3.upload(local_file=os.path.join(output_dir, "label_encoder_train.pkl"),path=os.path.join(train_s3_path, "outputs", "preprocess", "label_encoder_train.pkl"))

            # Ejecutamos las funciones de preprocesamiento test
            df_data_test_prepared  = prepare_dataset_test(test_clientes,test_requerimientos,df_imputaciones,artifact_encoders_clientes,artifact_scaler)
            df_data_test_prepared.to_csv(os.path.join(train_s3_path, "data", "out", "data_test_prepared.csv"), index=False)
            df_data_test_prepared.to_csv(os.path.join(output_dir, "data_test_prepared.csv"), index=False)
            mlflow.log_artifact(os.path.join(output_dir, "data_test_prepared.csv"), artifact_path="data/out")
    return run_id, data_pull_id