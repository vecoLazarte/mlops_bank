import pandas as pd
import fire
import os
from sklearn.model_selection import train_test_split


def buscar_indices_coincidentes(df_clientes,df_requerimientos):
    clientes_ids = set(df_clientes['ID_CORRELATIVO'])
    requerimientos_ids = set(df_requerimientos['ID_CORRELATIVO'])
    ids_comunes = list(clientes_ids.intersection(requerimientos_ids))
    clientes_filtrado = df_clientes[df_clientes['ID_CORRELATIVO'].isin(ids_comunes)].copy()
    requerimientos_filtrado = df_requerimientos[df_requerimientos['ID_CORRELATIVO'].isin(ids_comunes)].copy()
    return ids_comunes, clientes_filtrado, requerimientos_filtrado

def split_data(ids_comunes, perc_data_test):
    ids_train, ids_test = train_test_split(ids_comunes, test_size=perc_data_test, random_state=42)
    return ids_train, ids_test

def process_split_data():
    if (os.getcwd().endswith('src')):
        os.chdir("..")
    df_clientes = pd.read_csv("data/in/train_clientes_sample.csv")
    df_requerimientos = pd.read_csv("data/in/train_requerimientos_sample.csv")
    ids_comunes, clientes_filtrado, requerimientos_filtrado = buscar_indices_coincidentes(df_clientes, df_requerimientos)
    ids_train, ids_test = split_data(ids_comunes, 0.3)
    train_clientes = clientes_filtrado[clientes_filtrado['ID_CORRELATIVO'].isin(ids_train)]
    test_clientes = clientes_filtrado[clientes_filtrado['ID_CORRELATIVO'].isin(ids_test)]
    train_requerimientos = requerimientos_filtrado[requerimientos_filtrado['ID_CORRELATIVO'].isin(ids_train)]
    test_requerimientos = requerimientos_filtrado[requerimientos_filtrado['ID_CORRELATIVO'].isin(ids_test)]

    if (not (os.path.exists("data/out"))):
        os.mkdir("data/out")
    train_clientes.to_csv("data/out/clientes_data_train.csv", index=False)
    test_clientes.to_csv("data/out/clientes_data_test.csv", index=False)
    train_requerimientos.to_csv("data/out/requerimientos_data_train.csv", index=False)
    test_requerimientos.to_csv("data/out/requerimientos_data_test.csv", index=False)

def main():
    process_split_data()

if __name__ == "__main__":
    fire.Fire(main)
