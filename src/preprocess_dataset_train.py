import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import pickle
import mlflow
from mlflow.artifacts import download_artifacts

class PreprocessData:
    _output_path = ""

    def __init__(self, output_path):
        self._output_path = output_path
        self._create_output_path()

    def _create_output_path(self):
        if not(os.path.exists(self._output_path)):
            os.makedirs(self._output_path)

    def _save_y_col_name(self, y_col):
        df_y_col_name = pd.DataFrame({'y_col':[y_col]})
        df_y_col_name.to_csv(f'{self._output_path}/y_col_name.csv', index=False)
        mlflow.log_artifact(f'{self._output_path}/y_col_name.csv', artifact_path=f'{self._output_path}')

    def generar_variables_ingenieria(self,clientes_df):
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
    
    def imputacion_variables(self, clientes_df, requerimientos_df):
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
        df_imputaciones.to_csv(f'{self._output_path}/imputacion_parametros.csv', index=False)
        mlflow.log_artifact(f'{self._output_path}/imputacion_parametros.csv', artifact_path=f'{self._output_path}')

        return clientes_df, requerimientos_df

    
    def encoder_categoricos(self, clientes_df):
        clientes_df['RANG_SDO_PASIVO_MENOS0'] = clientes_df['RANG_SDO_PASIVO_MENOS0'].replace('Cero', 'Rango_SDO_00')
        clientes_df['FLAG_LIMA_PROVINCIA'] = clientes_df['FLAG_LIMA_PROVINCIA'].map({'Lima': 1, 'Provincia': 0})
        cat_cols = clientes_df.select_dtypes(include=['object', 'category']).columns
        encoders_clientes = {} 
        for col in cat_cols:
            le = LabelEncoder()
            clientes_df[col] = le.fit_transform(clientes_df[col])
            encoders_clientes[col] = le
        return clientes_df, encoders_clientes

    def construir_variables_requerimientos(self, df_reqs, id_col='ID_CORRELATIVO'):
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
    
    def estandarizacion(self, df_final):
        no_escalar = ['ID_CORRELATIVO', 'CODMES', 'ATTRITION']
        columnas_a_escalar = df_final.columns.difference(no_escalar)
        df_predictoras = df_final[columnas_a_escalar]
        scaler = StandardScaler()
        df_escaladas = pd.DataFrame(scaler.fit_transform(df_predictoras),columns=columnas_a_escalar,index=df_final.index)
        df_final_estandarizado = pd.concat([df_final[no_escalar], df_escaladas],axis=1)
        return df_final_estandarizado, scaler
    
    def _save_x_col_names(self, df_final, y_col):
        x_cols = [col for col in df_final.columns if col != y_col and col not in ['ID_CORRELATIVO', 'CODMES']]
        df_x_col_names = pd.DataFrame({'x_col': x_cols})
        df_x_col_names.to_csv(f'{self._output_path}/x_col_names.csv', index=False)
        mlflow.log_artifact(f'{self._output_path}/x_col_names.csv', artifact_path=f'{self._output_path}')

    def preprocess_dataset(self, clientes_df, requerimientos_df, y_col):
        self._save_y_col_name(y_col)
        clientes_df = self.generar_variables_ingenieria(clientes_df)
        clientes_df,requerimientos_df = self.imputacion_variables(clientes_df,requerimientos_df)
        clientes_df, artifact_encoders_clientes = self.encoder_categoricos(clientes_df)
        requerimientos_df = self.construir_variables_requerimientos(requerimientos_df)
        df_final = clientes_df.merge(requerimientos_df, on='ID_CORRELATIVO', how='inner')
        df_final, artifact_scaler = self.estandarizacion(df_final)
        self._save_x_col_names(df_final, y_col)
    
        return df_final, artifact_encoders_clientes, artifact_scaler

def process_preprocess_dataset(tracking_uri, experiment_name, parent_run_id, split_run_id):
    if (os.getcwd().endswith('src')):
        os.chdir("..")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name="preprocess_dataset_train", nested=True) as preprocess_train_run:
            preprocess_train_run_id = preprocess_train_run.info.run_id
            y_col = "ATTRITION"
            clientes_train_path = download_artifacts(run_id=split_run_id, artifact_path="data/out/clientes_data_train.csv")
            df_data_train = pd.read_csv(clientes_train_path)
            requerimiento_train_path = download_artifacts(run_id=split_run_id, artifact_path="data/out/requerimientos_data_train.csv")
            df_requerimientos_train = pd.read_csv(requerimiento_train_path)
            preprocess_data_instance = PreprocessData("outputs/preprocess")
            df_data_train_prepared, artifact_encoders_clientes, artifact_scaler = preprocess_data_instance.preprocess_dataset(df_data_train, df_requerimientos_train, y_col)
            df_data_train_prepared.to_csv("data/out/data_train_prepared.csv", index=False)
            mlflow.log_artifact('data/out/data_train_prepared.csv', artifact_path=f'data/out')
            with open('outputs/preprocess/scaler_train.pkl', 'wb') as f:
                pickle.dump(artifact_scaler, f)
            mlflow.log_artifact('outputs/preprocess/scaler_train.pkl', artifact_path="outputs/preprocess")
            with open('outputs/preprocess/label_encoder_train.pkl', 'wb') as f:
                pickle.dump(artifact_encoders_clientes, f)
            mlflow.log_artifact('outputs/preprocess/label_encoder_train.pkl', artifact_path="outputs/preprocess")
    return preprocess_train_run_id
