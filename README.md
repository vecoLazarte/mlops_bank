# Bank Attrition Detection MLOps
Construccion y despliegue de un modelo analítico que predice los clientes más propensos a fugar en los próximos 5 meses.

## 📌 Problemática

## 💡 Solución

### Pipeline de entrenamiento


### Pipeline de inferencia

#### Step: Data pull
Su objetivo principal es realizar la extracción, procesamiento y preparación de datos para el entrenamiento de modelos de machine learning, dejando todo listo en S3 y registrado en MLflow.

##### 🔧 Funcionalidad principal:

- Extracción de datos desde **AWS Athena** para clientes y requerimientos, con filtros de fecha basados en `cod_month`, `cod_month_start` y `cod_month_end`.
- Preprocesamiento completo, incluyendo:
  - Ingeniería de variables.
  - Imputación de valores faltantes (moda y mediana).
  - Codificación de variables categóricas mediante `LabelEncoder`.
  - Construcción de features agregados para requerimientos.
  - Estandarización con `StandardScaler`.
- División de datos en conjuntos de entrenamiento y prueba.
- Registro de artefactos en **MLflow** (`scaler`, `encoders`, columnas predictoras, parámetros de imputación, etc.).
- Almacenamiento de datasets preprocesados en **Amazon S3**.

#### 📁 Artefactos generados:

- `data_train_prepared.csv` y `data_test_prepared.csv`
- `scaler_train.pkl` y `label_encoder_train.pkl`
- `x_col_names.csv`, `y_col_name.csv`, `imputacion_parametros.csv`
- Datasets originales: `clientes_data_train.csv`, `clientes_data_test.csv`, `requerimientos_data_train.csv`, `requerimientos_data_test.csv`

#### Model training

#### Model Evaluation

#### Model Registration
