import fire
import pandas as pd
import os
import pickle


class PrepareData():
    _output_path = ""

    def __init__(self, output_path):
        self._output_path = output_path

    def prepare_impute_missing(self, df_data, x_cols):
        df_data_imputed = df_data.copy()
        df_impute_parameters = pd.read_csv(f"{self._output_path}/imputacion_parametros.csv")
        for col in x_cols:
            impute_value = df_impute_parameters[df_impute_parameters["variable"]==col]["valor"].values[0]
            df_data_imputed[col] = df_data_imputed[col].fillna(impute_value)
        return df_data_imputed
       
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
    
    def apply_label_encoders_to_test(self, df_test):
        df_test['RANG_SDO_PASIVO_MENOS0'] = df_test['RANG_SDO_PASIVO_MENOS0'].replace('Cero', 'Rango_SDO_00')
        df_test['FLAG_LIMA_PROVINCIA'] = df_test['FLAG_LIMA_PROVINCIA'].map({'Lima': 1, 'Provincia': 0})
        path_encoder=f"{self._output_path}/label_encoder_train.pkl"
        with open(path_encoder, 'rb') as f:
            encoders_clientes = pickle.load(f)
        for col, le in encoders_clientes.items():
            df_test[col] = le.transform(df_test[col])
        return df_test
    
    def aplicar_estandarizacion_test(self, df_test):
        path_scaler=f"{self._output_path}/scaler_train.pkl"
        with open(path_scaler, 'rb') as f:
            scaler = pickle.load(f)
        no_escalar = ['ID_CORRELATIVO', 'CODMES', 'ATTRITION']
        columnas_a_escalar = df_test.columns.difference(no_escalar)
        df_predictoras = df_test[columnas_a_escalar]
        df_escaladas = pd.DataFrame(scaler.transform(df_predictoras),columns=columnas_a_escalar,index=df_test.index)
        df_test_estandarizado = pd.concat([df_test[no_escalar], df_escaladas], axis=1)
        return df_test_estandarizado
    
    def prepare_dataset(self, df_data_test,df_requerimientos_test):
        x_cols_clientes = ['RANG_INGRESO','FLAG_LIMA_PROVINCIA','EDAD','ANTIGUEDAD']
        x_cols_requerimientos = ['DICTAMEN']
        df_data_imputed_clientes = self.prepare_impute_missing(df_data_test, x_cols_clientes)
        df_data_imputed_requerimientos = self.prepare_impute_missing(df_requerimientos_test, x_cols_requerimientos)
        df_data_feature_clientes = self.generar_variables_ingenieria(df_data_imputed_clientes)
        df_data_feature_requerimientos = self.construir_variables_requerimientos(df_data_imputed_requerimientos)
        df_data_encoder_clientes = self.apply_label_encoders_to_test(df_data_feature_clientes)
        df_final = df_data_encoder_clientes.merge(df_data_feature_requerimientos, on='ID_CORRELATIVO', how='inner')
        df_final = self.aplicar_estandarizacion_test(df_final)
        return df_final

def process_prepare_dataset():
    if (os.getcwd().endswith('src')):
        os.chdir("..")
    df_data_test = pd.read_csv("data/out/clientes_data_test.csv")
    df_requerimientos_test = pd.read_csv("data/out/requerimientos_data_test.csv")
    prepare_data_instance = PrepareData("outputs/preprocess")
    df_data_train_prepared  = prepare_data_instance.prepare_dataset(df_data_test,df_requerimientos_test)
    df_data_train_prepared.to_csv("data/out/data_test_prepared.csv", index=False)


def main():
    process_prepare_dataset()

if __name__ == "__main__":
    fire.Fire(main)
