import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

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

# 1. Tendencia Central
media_Sp = np.mean(div['Close'])
mediana_Sp = np.median(div['Close'])
moda_SP = pd.Series(div['Close']).mode().iloc[0]
desviacion_estandar_SP = np.std(div['Close'])
varianza_Sp = np.var(div['Close'])
rango_sp = np.max(div['Close']) - np.min(div['Close'])
frecuencia_absoluta_Sp = pd.Series(o['Close']).value_counts()
frecuencia_relativa_Sp = pd.Series(o['Close']).value_counts(normalize=True)

print(f"Media: {media_Sp}")
print(f"Mediana: {mediana_Sp}")
print(f"Moda: {moda_SP}")
print(f"desviación: {desviacion_estandar_SP}")
print(f"Varianza: {varianza_Sp}")
print(f"Rango: {rango_sp}")
print(f"frecuencia: {frecuencia_absoluta_Sp}")
print(f"frecuencia rela: {frecuencia_relativa_Sp}")


mediana_10 = np.median(dgs10_df['Precio_Bonos_10'])
mediana_west = np.median(wti_df['Precio_WTI'])
print(f"Mediana 10: {mediana_10}")
print(f"Mediana west: {mediana_west}")

#mediana_vol = pd.Series(vol['Close']).value_counts(normalize=True)
#mediana_bonos_1 = pd.Series(gs1m_df['Precio_Bonos_1']).value_counts(normalize=True)
#mediana_Bonos_2 = pd.Series(dgs2_df['Precio_Bonos_2']).value_counts(normalize=True)
#mediana_Bonos_10 = pd.Series(dgs10_df['Precio_Bonos_10']).value_counts(normalize=True)
#mediana_oro = pd.Series(o['Close']).value_counts().value_counts(normalize=True)
#mediana_DXY = pd.Series(DXY_FRED_df['DXY_FRED']).value_counts(normalize=True)
#mediana_wti = pd.Series(wti_df['Precio_WTI']).value_counts(normalize=True)
#mediana_HGY = pd.Series(yield_['Close']).value_counts(normalize=True)


#print(f"Mediana: {mediana_Sp}")
#print(f"Mediana: {mediana_vol}")
#print(f"Mediana: {mediana_bonos_1}")
#print(f"Mediana: {mediana_Bonos_2}")
#print(f"Mediana: {mediana_Bonos_10}")
#print(f"Mediana: {mediana_oro}")
#print(f"Mediana: {mediana_DXY}")
#print(f"Mediana: {mediana_wti}")
#print(f"Mediana: {mediana_HGY}")

# Imprimir los resultados
# print(f"Media: {media_Sp}")
#print(f"Media: {media_vol}")
#print(f"Media: {media_bonos_1}")
#print(f"Media: {media_Bonos_2}")
#print(f"Media: {media_Bonos_10}")
#print(f"Media: {media_oro}")
#print(f"Media: {media_DXY}")
#print(f"Media: {media_wti}")
#print(f"Media: {media_HGY}")


