# Bank Attrition Detection MLOps
Construccion y despliegue de un modelo analítico que predice los clientes más propensos a fugar en los próximos 5 meses.

## 📌 Problemática

## 💡 Solución

### Pipeline de entrenamiento

🚀 Detalles técnicos:
- Ejecutado en una instancia `ml.m5.2xlarge` con contenedor personalizado (`image_uri` de ECR).
- Integrado con MLflow Tracking Server definido por `TRACKING_SERVER_ARN`.

#### Step: Data pull
Su objetivo principal es realizar la extracción, procesamiento y preparación de datos para el entrenamiento de modelos de machine learning, dejando todo listo en S3 y registrado en MLflow.

🔧 Funcionalidad principal:

- Extracción de datos desde **AWS Athena** para clientes y requerimientos, con filtros de fecha basados en `cod_month`, `cod_month_start` y `cod_month_end`.
- Preprocesamiento completo, incluyendo:
  - Ingeniería de variables.
  - Imputación de valores faltantes.
  - Codificación de variables categóricas mediante `LabelEncoder`.
  - Construcción de features agregados para requerimientos.
  - Estandarización con `StandardScaler`.
- División de datos en conjuntos de entrenamiento y prueba.

📁 Artefactos almacenados en **MLflow** y **Amazon S3**:

- `data_train_prepared.csv` y `data_test_prepared.csv`
- `scaler_train.pkl` y `label_encoder_train.pkl`
- `x_col_names.csv`, `y_col_name.csv`, `imputacion_parametros.csv`
- Datasets originales: `clientes_data_train.csv`, `clientes_data_test.csv`, `requerimientos_data_train.csv`, `requerimientos_data_test.csv`

#### Step: Model training

Su propósito es ejecutar entrenamiento y validación cruzada de modelos usando datos preprocesados previamente, registrando todos los resultados y artefactos en MLflow y almacenándolos en S3.

🔧 Funcionalidad principal:
- Descarga de datasets y artefactos generados en el `step` de `data_pull`, incluyendo:
  - Columnas predictoras (`x_col_names.csv`)
  - Columna objetivo (`y_col_name.csv`)
  - Dataset de entrenamiento (`data_train_prepared.csv`)
- Configuración de hiperparámetros para los modelos:
  - `XGBClassifier` con búsqueda en grilla (GridSearchCV)
  - `RandomForestClassifier` con búsqueda en grilla (GridSearchCV)
- Entrenamiento con validación cruzada (CV) y evaluación utilizando la métrica `roc_auc`.  
- Registro de métricas y parámetros en MLflow:
  - Parámetros óptimos encontrados en la búsqueda de hiperparámetros.
  - Métricas de CV: media, desviación estándar y coeficiente de variación (`auc_score_cv`).

📁 Artefactos almacenados en **MLflow** y **Amazon S3**:
- `train_cv_model_results.csv` y `train_cv_model_results_best_model.csv`: resultados completos y del mejor modelo por algoritmo.
- `feature_importance.csv`: ranking de importancia de variables para cada modelo.
- `grid_search_model.pickle`: modelo entrenado serializado (uno por algoritmo).

#### Step: Model Evaluation

#### Step: Model Registration

### Pipeline de inferencia
