# Bank Attrition Detection MLOps
Construccion y despliegue de un modelo analítico que predice los clientes más propensos a fugar en los próximos 5 meses.

## 📌 Problemática

## 💡 Solución

### ⚙️ GitHub Actions: CI/CD for Build and Push Docker to ECR

Este workflow automatiza el proceso de construcción y despliegue de una imagen Docker personalizada en Amazon ECR, lista para ser utilizada en el pipeline de entrenamiento con Amazon SageMaker.

#### 🧬 Flujo del workflow

1. 🚀 Se lanza una máquina virtual con Ubuntu como runner.
2. 🔐 Se configuran las credenciales de AWS.
3. 🐳 Se inicia sesión en Amazon ECR.
4. 🏗️ Se construye una imagen Docker personalizada que incluye:
   - Imagen base oficial: `python:3.9`
   - Instalación de librerías esenciales para ciencia de datos y MLOps:
     - `mlflow`, `sagemaker`, `sagemaker-mlflow`
     - `xgboost`, `boto3`, `pandas`, `numpy`
     - `awswrangler`, `fsspec`, `s3fs`
5. 🏷️ Se aplica un tag a la imagen.
6. 📤 Se publica la imagen en el repositorio correspondiente de Amazon ECR.


### 🧪 Pipeline de entrenamiento

- Integrado con **MLflow Tracking Server**.
- Ejecutado dentro de un contenedor creado por el workflow CI/CD, utilizando una instancia `ml.m5.2xlarge`.

### ⚙️ GitHub Actions: CI/CD for Training Pipeline

Este workflow crea y ejecuta el pipeline de entrenamiento mediante los siguientes pasos:

1. 🚀 Se lanza una máquina virtual con Ubuntu como runner.
2. 🔐 Se configuran las credenciales de AWS.
3. 🏗️ Se construye un pipeline de entrenamiento dentro de SageMaker considerando los siguientes steps: Data Pull, Model Training, Model Evaluation y Model Registration.

#### 🧾 Step: Data Pull

Realiza la extracción, procesamiento y preparación de datos para el entrenamiento de modelos, dejando todo listo en S3 y registrado en MLflow.

##### 🔧 Funcionalidad principal

- Extracción de datos desde **AWS Athena** (clientes y requerimientos), aplicando filtros de fecha: `cod_month`, `cod_month_start`, `cod_month_end`.
- Preprocesamiento completo:
  - Ingeniería de variables.
  - Imputación de valores faltantes.
  - Codificación con `LabelEncoder`.
  - Agregación de features por requerimientos.
  - Estandarización con `StandardScaler`.
- División en conjuntos de entrenamiento y prueba.

##### 📁 Artefactos almacenados

En **MLflow** y **Amazon S3**:

- `data_train_prepared.csv`, `data_test_prepared.csv`
- `scaler_train.pkl`, `label_encoder_train.pkl`
- `x_col_names.csv`, `y_col_name.csv`, `imputacion_parametros.csv`
- Datasets originales:
  - `clientes_data_train.csv`, `clientes_data_test.csv`
  - `requerimientos_data_train.csv`, `requerimientos_data_test.csv`

#### 🤖 Step: Model Training

Entrenamiento y validación cruzada de modelos con datos procesados, registrando todo en MLflow y almacenándolo en S3.

##### 🔧 Funcionalidad principal

- Descarga de artefactos generados en `data_pull`:
  - `x_col_names.csv`, `y_col_name.csv`, `data_train_prepared.csv`
- Configuración y entrenamiento de modelos:
  - `XGBClassifier` y `RandomForestClassifier` con GridSearchCV
- Validación cruzada con métrica `roc_auc`
- Registro en MLflow:
  - Parámetros óptimos y métricas estadísticas (`auc_score_cv`)

##### 📁 Artefactos almacenados

En **MLflow** y **Amazon S3**:

- `train_cv_model_results.csv`, `train_cv_model_results_best_model.csv`
- `feature_importance.csv`
- `grid_search_model.pickle`

#### Step: Model Evaluation

#### Step: Model Registration

### Pipeline de inferencia
