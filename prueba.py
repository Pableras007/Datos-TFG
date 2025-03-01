import yfinance as yf
from fredapi import Fred

# API Key
fred = Fred(api_key="53066d54b858b20a36f134b6be373445")

# Definir los tickers
sp500tr = yf.Ticker("^SP500TR")
vix = yf.Ticker("^VIX")
oro = yf.Ticker("GC=F")
dxy = yf.Ticker("DX-Y.NYB")
baltic = yf.Ticker("BDRY")
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
dry = baltic.history(period="max")
yield_ = hyg.history(period="max")

# Almacenar datos en archivos CSV
hist.to_csv("max_sp500tr.csv")
vol.to_csv("max_volatilidad.csv")
o.to_csv("max_oro.csv")
div.to_csv("max_dxy.csv")
yield_.to_csv("max_hyg.csv")
wti_df.to_csv('max_wti_crude_oil_prices.csv', index_label='Fecha')
dgs10_df.to_csv('max_dgs10.csv', index_label='Fecha')
dgs2_df.to_csv('max_dgs2.csv', index_label='Fecha')
gs1m_df.to_csv('max_gs1m.csv', index_label='Fecha')

DXY_FRED_df.to_csv('max_DXY_FRED.csv', index_label='Fecha')
dry.to_csv("max_baltic.csv")

# Mostrar los primeros 5 registros de cada DataFrame
print(hist.head())
print(vol.head())
print(o.head())
print(div.head())
print(dry.head())
print(yield_.head())

