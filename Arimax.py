import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Cargar y limpiar cada archivo CSV y seleccionar solo la columna 'Close'
hist = pd.read_csv('max_sp500tr.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
vol = pd.read_csv('max_volatilidad.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
o = pd.read_csv('max_oro.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
div = pd.read_csv('max_dxy.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
yield_ = pd.read_csv('max_hyg.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']

# Cargar datos FRED desde CSV (sin cambios en la estructura de datos de FRED)
wti_df = pd.read_csv('max_wti_crude_oil_prices.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_WTI'})
dgs10_df = pd.read_csv('max_dgs10.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_Bonos_10'})
dgs2_df = pd.read_csv('max_dgs2.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_Bonos_2'})
gs1m_df = pd.read_csv('max_gs1m.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_Bonos_1'})

# Asegurarse de que los índices sean datetime y en UTC
hist.index = pd.to_datetime(hist.index, utc=True)
vol.index = pd.to_datetime(vol.index, utc=True)
o.index = pd.to_datetime(o.index, utc=True)
yield_.index = pd.to_datetime(yield_.index, utc=True)
div.index = pd.to_datetime(div.index, utc=True)

# Asegurarse de que los índices de FRED sean compatibles
gs1m_df.index = pd.to_datetime(gs1m_df.index, utc=True)
dgs2_df.index = pd.to_datetime(dgs2_df.index, utc=True)
dgs10_df.index = pd.to_datetime(dgs10_df.index, utc=True)
wti_df.index = pd.to_datetime(wti_df.index, utc=True)

# Eliminar la zona horaria y mantener solo la fecha (año-mes-día)
hist.index = hist.index.date
vol.index = vol.index.date
o.index = o.index.date
yield_.index = yield_.index.date
div.index = div.index.date

# Asegurarse de que los índices de FRED sean compatibles
gs1m_df.index = gs1m_df.index.date
dgs2_df.index = dgs2_df.index.date
dgs10_df.index = dgs10_df.index.date
wti_df.index = wti_df.index.date

# Combinar todas las series temporales en un DataFrame unificado
combined_df = pd.DataFrame({
    'SP500': hist,
    'Volatilidad': vol,
    'Bonos 10 Años': dgs10_df['Precio_Bonos_10'],
    'Oro': o,
    'DXY': div,
    'WTI': wti_df['Precio_WTI'],
})

# Reindexar para tener un rango temporal específico
date_range = pd.date_range(start='2000-08-30', end='2025-01-17', freq='D')
combined_df.index = pd.to_datetime(combined_df.index)
combined_df = combined_df.reindex(date_range)

# Rellenar valores nulos con el precio del día anterior
combined_df = combined_df.fillna(method='ffill')

# Función para realizar la prueba ADF y verificar la estacionariedad
def realizar_adf(serie):
    resultado = adfuller(serie)
    print(f"Estadístico ADF: {resultado[0]}")
    print(f"Valor p: {resultado[1]}")
    print(f"Valor crítico (1%): {resultado[4]['1%']}")
    return resultado[1]  # Retorna el valor p

# Aplicar la prueba ADF a la serie objetivo 'SP500'
target_series = combined_df['SP500']
target_p_value = realizar_adf(target_series)

# Aplicar la prueba ADF a las exógenas y diferenciarlas si no son estacionarias
exog_vars = combined_df[['Volatilidad', 'Bonos 10 Años', 'Oro', 'DXY', 'WTI']]

for col in exog_vars.columns:
    p_value = realizar_adf(exog_vars[col])
    if p_value > 0.05:
        print(f"La variable {col} no es estacionaria y necesita ser diferenciada.")
        exog_vars[col] = exog_vars[col].diff().dropna()  # Diferenciar si no es estacionaria

# Asegurarse de que las fechas de las exógenas están alineadas con la serie objetivo
exog_vars = exog_vars.loc[target_series.index]

# Reemplazar los valores infinitos con NaN y rellenar los valores nulos
exog_vars = exog_vars.replace([np.inf, -np.inf], np.nan)  # Reemplazar inf por NaN
exog_vars = exog_vars.fillna(method='bfill')  # Rellenar valores NaN con el anterior

# Ajustar el modelo ARIMAX
model_arimax = ARIMA(target_series, exog=exog_vars, order=(2, 1, 2))
results_arimax = model_arimax.fit()

# Resumen del modelo
print(results_arimax.summary())

# Predicción
forecast = results_arimax.forecast(steps=30, exog=exog_vars.tail(30))

# Visualizar las predicciones y los resultados reales
plt.figure(figsize=(12, 6))
plt.plot(target_series, label='Datos reales', color='blue')
plt.plot(forecast.index, forecast, label='Pronóstico ARIMAX', color='red')
plt.legend()

# Zoom en la predicción (últimos 30 días)
plt.xlim(forecast.index[0] - pd.Timedelta(days=5), forecast.index[-1] + pd.Timedelta(days=5))  # Enfocar en los últimos 30 días
plt.title('Predicción ARIMAX vs Datos Reales (últimos 30 días)')
plt.xlabel('Fecha')
plt.ylabel('Valor del SP500')
plt.show()

# Medir el rendimiento (RMSE)
y_true = target_series.tail(30)  # Últimos 30 valores reales
rmse = np.sqrt(mean_squared_error(y_true, forecast))  # Calcular RMSE
print(f"RMSE (Error cuadrático medio de la raíz): {rmse}")
