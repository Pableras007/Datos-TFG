import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# API Key
fred = Fred(api_key="53066d54b858b20a36f134b6be373445")

# Definir los tickers
sp500tr = yf.Ticker("^SP500TR")
vix = yf.Ticker("^VIX")
oro = yf.Ticker("GC=F")
dxy = yf.Ticker("DX-Y.NYB")
hyg = yf.Ticker("HYG")

# Obtener datos de FRED
wti_data = fred.get_series('DCOILWTICO', observation_start='1950-01-01')  # Iniciar desde 1950
dgs10 = fred.get_series('DGS10', observation_start='1950-01-01')
dgs2 = fred.get_series('GS2', observation_start='1950-01-01')
gs1m = fred.get_series('GS1M', observation_start='1950-01-01')

# Convertir datos de FRED a DataFrames
wti_df = wti_data.to_frame(name='Precio_WTI')
dgs10_df = dgs10.to_frame(name='Precio_Bonos_10')
dgs2_df = dgs2.to_frame(name='Precio_Bonos_2')
gs1m_df = gs1m.to_frame(name='Precio_Bonos_1')


# Descargar datos históricos de Yahoo Finance
hist = sp500tr.history(period="max")
vol = vix.history(period="max")
o = oro.history(period="max")
div = dxy.history(period="max")
yield_ = hyg.history(period="max")

# Asegurarse de que todos los índices no tengan zona horaria
hist.index = hist.index.tz_localize(None)
vol.index = vol.index.tz_localize(None)
o.index = o.index.tz_localize(None)
yield_.index = yield_.index.tz_localize(None)
div.index = div.index.tz_localize(None)

# Asegurarse de que los índices de FRED sean compatibles
gs1m_df.index = gs1m_df.index.tz_localize(None)
dgs2_df.index = dgs2_df.index.tz_localize(None)
dgs10_df.index = dgs10_df.index.tz_localize(None)
wti_df.index = wti_df.index.tz_localize(None)

# 2. Crear un DataFrame combinando las series
# Asegúrate de alinear los índices
data = pd.DataFrame({
    'SP500': hist['Close'],
    'Volatilidad': vol['Close'],
    'Bonos 1 Mes': gs1m_df['Precio_Bonos_1'],
    'Bonos 2 Años': dgs2_df['Precio_Bonos_2'],
    'Bonos 10 Años': dgs10_df['Precio_Bonos_10'],
    'Oro': o['Close'],
    'DXY': div['Close'],
    'WTI': wti_df['Precio_WTI'],
    'High Yield': yield_['Close']
})

# Crear boxplots individuales para cada variable
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

for ax, column in zip(axes.flatten(), data.columns):
    sns.boxplot(data=data[column], ax=ax)
    ax.set_title(f'Distribución de {column}')
    ax.grid(True)

plt.tight_layout()
plt.show()


# 3. Tendencias
# Normalizar datos
data_normalized = (data - data.min()) / (data.max() - data.min())
# Filtrar los datos desde el año 2000
data_filtered = data_normalized.loc[data_normalized.index >= '2000-01-01']

# Graficar las tendencias filtradas
plt.figure(figsize=(12, 6))

plt.plot(data_filtered.index, data_filtered['SP500'], label='SP500')
plt.plot(data_filtered.index, data_filtered['WTI'], label='WTI')
plt.plot(data_filtered.index, data_filtered['Oro'], label='Oro')
plt.plot(data_filtered.index, data_filtered['DXY'], label='DXY')
plt.plot(data_filtered.index, data_filtered['High Yield'], label='High Yield')
plt.plot(data_filtered.index, data_filtered['Bonos 10 Años'], label='Bonos 10 Años')

plt.title('Tendencias de Variables Seleccionadas (Normalizadas) desde el Año 2000')
plt.xlabel('Tiempo')
plt.ylabel('Valor Normalizado')
plt.legend()
plt.grid(True)
plt.show()


# 4. Calcular correlaciones entre las variables
correlation_matrix = data.corr()

# Heatmap con Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Correlación entre Variables')
plt.show()

# Función para descomponer una serie y evaluar estacionariedad
def descomponer_y_analizar_series(data):
    fig_trend, axes_trend = plt.subplots(3, 3, figsize=(18, 12))
    fig_seasonal, axes_seasonal = plt.subplots(3, 3, figsize=(18, 12))
    fig_resid, axes_resid = plt.subplots(3, 3, figsize=(18, 12))
    fig_diff, axes_diff = plt.subplots(3, 3, figsize=(18, 12))  # Gráficos separados para la diferenciación

    axes_trend = axes_trend.flatten()
    axes_seasonal = axes_seasonal.flatten()
    axes_resid = axes_resid.flatten()
    axes_diff = axes_diff.flatten()  # Aplanar para poder usar un índice

    # Diferenciación conjunta
    for idx, columna in enumerate(data.columns):
        serie = data[columna].ffill().dropna()

        if len(serie) >= 24:
            # Descomposición clásica
            descomposicion = seasonal_decompose(serie, model='additive', period=12)

            # Tendencia
            descomposicion.trend.plot(ax=axes_trend[idx], title=f'Tendencia de {columna}')
            axes_trend[idx].grid(True)

            # Estacionalidad
            descomposicion.seasonal.plot(ax=axes_seasonal[idx], title=f'Estacionalidad de {columna}')
            axes_seasonal[idx].grid(True)

            # Residuo
            descomposicion.resid.plot(ax=axes_resid[idx], title=f'Residuo de {columna}')
            axes_resid[idx].grid(True)
        else:
            axes_trend[idx].text(0.5, 0.5, f'{columna} insuficiente para descomposición',
                                 fontsize=10, ha='center', va='center')
            axes_seasonal[idx].axis('off')
            axes_resid[idx].axis('off')

        # Prueba de Dickey-Fuller
        resultado_adf = adfuller(serie)
        print(f"\n=== Análisis de la serie temporal: {columna} ===")
        print(f"Estadístico ADF: {resultado_adf[0]:.4f}")
        print(f"p-valor: {resultado_adf[1]:.4f}")
        print("La serie es estacionaria." if resultado_adf[1] < 0.05 else "La serie NO es estacionaria.")

        # Diferenciación conjunta
        serie_diferenciada = serie.diff().dropna()
        axes_diff[idx].plot(serie_diferenciada, label=f'{columna} Diferenciado')
        axes_diff[idx].set_title(f'Diferenciación de {columna}')
        axes_diff[idx].grid(True)
        axes_diff[idx].legend()

    # Ajustar gráficos
    fig_trend.tight_layout()
    fig_seasonal.tight_layout()
    fig_resid.tight_layout()
    fig_diff.tight_layout()  # Asegurarse de ajustar los gráficos de diferenciación

    # Mostrar gráficos
    plt.show()

# Aplicar la función al DataFrame
descomponer_y_analizar_series(data)
