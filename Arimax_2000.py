import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Cargar y limpiar cada archivo CSV y seleccionar solo la columna 'Close'
hist = pd.read_csv('max_sp500tr.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
vol = pd.read_csv('max_volatilidad.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
o = pd.read_csv('max_oro.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
div = pd.read_csv('max_dxy.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']
yield_ = pd.read_csv('max_hyg.csv', parse_dates=['Date'], index_col='Date').rename(columns={'Value': 'Close'})['Close']

# Cargar datos FRED desde CSV
wti_df = pd.read_csv('max_wti_crude_oil_prices.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_WTI'})
dgs10_df = pd.read_csv('max_dgs10.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_Bonos_10'})
dgs2_df = pd.read_csv('max_dgs2.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_Bonos_2'})
gs1m_df = pd.read_csv('max_gs1m.csv', parse_dates=['Fecha'], index_col='Fecha').rename(columns={'Value': 'Precio_Bonos_1'})

# Asegurar que las fechas estén alineadas
hist.index = pd.to_datetime(hist.index, utc=True).date
vol.index = pd.to_datetime(vol.index, utc=True).date
o.index = pd.to_datetime(o.index, utc=True).date
yield_.index = pd.to_datetime(yield_.index, utc=True).date
div.index = pd.to_datetime(div.index, utc=True).date

# FRED data
gs1m_df.index = pd.to_datetime(gs1m_df.index, utc=True).date
dgs2_df.index = pd.to_datetime(dgs2_df.index, utc=True).date
dgs10_df.index = pd.to_datetime(dgs10_df.index, utc=True).date
wti_df.index = pd.to_datetime(wti_df.index, utc=True).date

# Combinar todas las series temporales
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

# Rellenar valores nulos
combined_df = combined_df.fillna(method='ffill')

# Crear variable binaria de caídas (downfalls) -> 1 si la caída diaria es mayor a -1%
combined_df['Return'] = combined_df['SP500'].pct_change()
combined_df['Downfall'] = (combined_df['Return'] < -0.01).astype(int)

# Variables exógenas
exog_vars = combined_df[['Volatilidad', 'Bonos 10 Años', 'Oro', 'DXY', 'WTI']]

# Alinear exógenas y target
exog_vars = exog_vars.loc[combined_df.index]

# Test de Dickey-Fuller para verificar estacionariedad y diferenciación si es necesario
def dickey_fuller_and_diff(series):
    adf_result = adfuller(series.dropna())
    print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")

    # Si la serie no es estacionaria, diferenciamos
    if adf_result[1] > 0.05:
        print("No es estacionaria. Aplicando diferenciación...")
        return series.diff().dropna()  # Aplica diferenciación y elimina valores NaN generados
    else:
        print("La serie ya es estacionaria.")
        return series

# Aplicar el test Dickey-Fuller y diferenciación
combined_df['SP500'] = dickey_fuller_and_diff(combined_df['SP500'])
exog_vars['Volatilidad'] = dickey_fuller_and_diff(exog_vars['Volatilidad'])
exog_vars['Bonos 10 Años'] = dickey_fuller_and_diff(exog_vars['Bonos 10 Años'])
exog_vars['Oro'] = dickey_fuller_and_diff(exog_vars['Oro'])
exog_vars['DXY'] = dickey_fuller_and_diff(exog_vars['DXY'])
exog_vars['WTI'] = dickey_fuller_and_diff(exog_vars['WTI'])

# Dividir en train (70%) y test (30%)
split_point = int(len(combined_df) * 0.7)
train_data = combined_df.iloc[:split_point]
test_data = combined_df.iloc[split_point:]

# Dividir exógenas
train_exog = exog_vars.iloc[:split_point]
test_exog = exog_vars.iloc[split_point:]

# Reemplazar los valores infinitos con NaN y rellenar los valores nulos
train_exog = train_exog.replace([np.inf, -np.inf], np.nan)  # Reemplazar inf por NaN
train_exog = train_exog.fillna(method='bfill')  # Rellenar valores NaN con el siguiente valor en la serie

test_exog = test_exog.replace([np.inf, -np.inf], np.nan)
test_exog = test_exog.fillna(method='bfill')

# Ajuste ARIMAX
model_arimax = ARIMA(train_data['Downfall'], exog=train_exog, order=(2, 1, 2))
results_arimax = model_arimax.fit()

# Predicciones (probabilidades)
forecast_prob = results_arimax.predict(exog=test_exog, start=test_data.index[0], end=test_data.index[-1])

# Convertir probabilidades a clases binarias (umbral del 60%)
forecast_class = (forecast_prob > 0.6).astype(int)

# Asegurarse de que las predicciones tengan el mismo tamaño que el conjunto de prueba
forecast_class.index = test_data.index

# Evaluar modelo
print("Classification Report:")
print(classification_report(test_data['Downfall'], forecast_class))
print("ROC AUC Score:", roc_auc_score(test_data['Downfall'], forecast_prob))

# Matriz de confusión
conf_matrix = confusion_matrix(test_data['Downfall'], forecast_class)
print("Confusion Matrix:")
print(conf_matrix)

# Visualización de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho: No Caída', 'Predicho: Caída'], yticklabels=['Real: No Caída', 'Real: Caída'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()


# Seleccionar los datos entre febrero y mayo de 2020
start_date = '2020-02-01'
end_date = '2020-05-31'

# Extraer los datos reales y las predicciones para ese intervalo
selected_data = test_data[start_date:end_date]
selected_forecast_class = forecast_class[start_date:end_date]

# Visualizar la predicción vs los valores reales durante este período
plt.figure(figsize=(12, 6))
plt.plot(selected_data.index, selected_data['Downfall'], label='Real', color='blue')
plt.plot(selected_forecast_class.index, selected_forecast_class, label='Predicho', color='red', linestyle='--')
plt.title(f'Comparación de Caídas Reales vs Predichas - Febrero 2020 a Mayo 2020')
plt.legend()
plt.show()
