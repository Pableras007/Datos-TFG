import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

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

#outliers
# Calcular Q1, Q3 y IQR
Q1 = data.quantile(0.25)  # 25%
Q3 = data.quantile(0.75)  # 75%
IQR = Q3 - Q1

# Definir límites
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar outliers
outliers = (data < lower_bound) | (data > upper_bound)

# Mostrar resultados
print("Outliers detectados:")
print(outliers.sum())  # Número de outliers por variable

from scipy.stats import zscore

# Calcular Z-Scores
z_scores = data.apply(zscore)

# Identificar outliers (|Z| > 3)
outliers_z = (z_scores.abs() > 3)

# Mostrar resultados
print("Outliers detectados con Z-Score:")
print(outliers_z.sum())  # Número de outliers por variable

outlier_percentage = (outliers.sum() / len(data)) * 100
print("Porcentaje de outliers por variable:\n", outlier_percentage)
# Filtrar solo las columnas 'Date' y 'Close' en cada DataFrame
hist = hist[['Close']]
vol = vol[['Close']]
o = o[['Close']]
yield_ = yield_[['Close']]
div = div[['Close']]

# Asegúrate de que la columna 'Date' esté configurada como índice si no lo está
hist.reset_index(inplace=True)
vol.reset_index(inplace=True)
o.reset_index(inplace=True)
yield_.reset_index(inplace=True)
div.reset_index(inplace=True)

# Función para contar valores nulos antes y después de rellenarlos

def contar_y_reemplazar_nulos(df):
    # Contar los valores nulos antes del reemplazo
    nulos_antes = df.isnull().sum()
    print("\nNúmero de valores nulos por columna antes del reemplazo:")
    print(nulos_antes)

    # Reemplazar los valores nulos por el valor del día anterior
    df_filled = df.fillna(method='ffill')

    # Contar los valores nulos después del reemplazo
    nulos_despues = df_filled.isnull().sum()
    print("\nNúmero de valores nulos por columna después del reemplazo:")
    print(nulos_despues)

    return df_filled

# Aplicar la función a cada DataFrame
wti_df = contar_y_reemplazar_nulos(wti_df)
dgs10_df = contar_y_reemplazar_nulos(dgs10_df)
dgs2_df = contar_y_reemplazar_nulos(dgs2_df)
gs1m_df = contar_y_reemplazar_nulos(gs1m_df)


hist = contar_y_reemplazar_nulos(hist)
vol = contar_y_reemplazar_nulos(vol)
o = contar_y_reemplazar_nulos(o)
yield_ = contar_y_reemplazar_nulos(yield_)
div = contar_y_reemplazar_nulos(div)
