# Bank Attrition Detection MLOps
Construccion y despliegue de un modelo analítico que predice los clientes más propensos a fugar en los próximos 5 meses.

## Creacion de variables: 

##### SDO_ACTIVO
- Se crean las variables VAR_SDO_ACTIVO_6M y VAR_SDO_ACTIVO_3M con el objetivo de capturar diferentes tipos de comportamiento en la evolución del nivel de deuda.
    - VAR_SDO_ACTIVO_6M refleja la tendencia general del endeudamiento proyectado a 6 meses, útil para identificar patrones sostenidos.
    - VAR_SDO_ACTIVO_3M compara el promedio de deuda de los últimos 3 meses frente a los 3 meses anteriores, permitiendo detectar cambios abruptos recientes que podrían indicar un mayor riesgo de fuga.
        - Si es positiva → la deuda aumentará → posible riesgo de fuga si está sobreendeudado.
        - Si es negativa → la deuda disminuirá → comportamiento posiblemente saludable.
        - Si es cero → no hay cambio.

- Se crea la variable 'PROM_SDO_ACTIVO6M' con el fin de resumir el nivel promedio de endeudamiento durante los próximos 6 meses. Esta métrica proporciona una visión general y estable del comportamiento financiero del cliente, permitiendo capturar la magnitud total de su deuda en el mediano plazo.

- Se crea la variable 'PROM_SDO_ACTIVO_0M_2M' que representa el promedio del saldo de deuda para los próximos 3 meses (meses 0, 1 y 2). Esta variable permite evaluar el comportamiento actual o reciente del cliente, útil para detectar señales tempranas de riesgo.

- Se crea la variable 'PROM_SDO_ACTIVO_3M_5M' que calcula el promedio del saldo de deuda en el periodo de los meses 3 a 5. Esta métrica funciona como punto de comparación para identificar cambios de tendencia respecto al trimestre más reciente.

##### FLG_SEGURO
- Se crea la variable MESES_CON_SEGURO con el objetivo de identificar el número de meses, dentro del periodo de análisis, en los que el cliente contó con un seguro activo. Esta variable permite medir la consistencia en el uso de productos complementarios, como los seguros, que pueden estar asociados a una mayor vinculación con la entidad financiera.

##### NRO_ACCES_CANAL
- Se crea la variable 'VAR_NRO_ACCES_CANAL{n}_6M' que calcula la variación total en el número de accesos al canal {n} en un periodo de 6 meses. Su objetivo es detectar una tendencia global en el uso del canal, lo que puede reflejar mayor o menor vinculación del cliente con la entidad.
- Se crea la variable 'PROM_NRO_ACCES_CANAL{n}_6M' que representa el promedio de accesos al canal {n} en los próximos 6 meses, lo que permite medir el nivel general de uso del canal en el periodo de 6 meses.
Una mayor frecuencia puede estar asociada a una mayor interacción o fidelidad, mientras que un bajo promedio podría indicar desconexión o riesgo de fuga.
- Se crea la variable 'PROM_NRO_ACCES_CANAL{n}_0M_2M' que calcula el promedio de accesos recientes al canal {n} durante los primeros 3 meses (mes 0 al mes 2). Es útil para capturar el comportamiento actual del cliente, permitiendo detectar señales tempranas de desapego o aumento de interacción.
- Se crea la variable 'PROM_NRO_ACCES_CANAL{n}_3M_5M' que representa el promedio de accesos al canal {n} en el trimestre siguiente (meses 3 a 5). Se utiliza como punto de comparación para evaluar cambios recientes en el comportamiento frente al promedio anterior.
- Se crea la variable 'VAR_NRO_ACCES_CANAL{n}_3M' que mide la variación trimestral en el uso del canal {n}, comparando el promedio del trimestre (meses 3 a 5) con el del trimestre (meses 0 a 2). Una caída en el uso puede ser una señal de desconexión progresiva del cliente, mientras que un aumento puede indicar mayor involucramiento.

##### NRO_ENTID_SSFF
- Se crea la variable 'PROM_NRO_ENTID_SSFF' que calcula el promedio del número de entidades del sistema financiero (SSFF) con las que el cliente ha mantenido relación durante los próximos 6 meses. Esta variable resume el nivel general de multibancarización en el periodo analizado y permite detectar clientes potencialmente menos fidelizados.
- Se crea la variable 'VAR_NRO_ENTID_SSFF_6M' que mide la variación total en el número de entidades financieras con las que el cliente interactúa, entre el mes actual y el sexto mes. Un valor positivo sugiere que el cliente aumentará su multibancarización (posible señal de fuga), mientras que un valor negativo puede indicar consolidación financiera.
- Se crea la variable 'PROM_NRO_ENTID_SSFF_0M_2M' que representa el promedio reciente de entidades SSFF con las que el cliente interactúa, considerando los meses 0 al 2. Permite observar el comportamiento actual del cliente respecto a su nivel de diversificación financiera.
- Se crea la variable 'PROM_NRO_ENTID_SSFF_3M_5M' que calcula el promedio del trimestre siguiente (meses 3 a 5) en relación con el número de entidades SSFF. Funciona como punto de comparación para evaluar cambios recientes.
- Se crea la variable 'VAR_NRO_ENTID_SSFF_3M' que mide la variación en el número promedio de entidades entre los dos trimestres analizados.
    - Si el valor es positivo, el cliente aumentó su nivel de multibancarización (posible fuga).
    - Si es negativo, el cliente disminuyó su interacción con otras entidades (mayor fidelidad).

##### FLG_SDO_OTSSFF_MENOS
- Se crea la variable 'MESES_CON_SALDO' con el objetivo de contabilizar el número de meses, dentro del periodo analizado, en los que el cliente ha mantenido saldo en otras entidades del sistema financiero (SSFF).
Esta variable permite evaluar el grado de diversificación financiera activa del cliente, lo cual puede estar relacionado con un mayor riesgo de fuga si se observa una presencia frecuente de saldos fuera del banco principal.

