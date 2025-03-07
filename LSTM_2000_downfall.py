import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, mean_squared_error
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.optimizers import Adam


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

# Rellenar fechas y valores faltantes
date_range = pd.date_range(start='2000-08-30', end='2025-01-17', freq='D')
combined_df.index = pd.to_datetime(combined_df.index)
combined_df = combined_df.reindex(date_range).fillna(method='ffill')

# Variables adicionales
combined_df['Return'] = combined_df['SP500'].pct_change()
combined_df['Downfall'] = (combined_df['Return'] < -0.01).astype(int)
combined_df['SP500_MA_7'] = combined_df['SP500'].rolling(window=7).mean()
combined_df['SP500_MA_30'] = combined_df['SP500'].rolling(window=30).mean()
combined_df = combined_df.fillna(method='bfill')

# Visualizar balance de clases
print(combined_df['Downfall'].value_counts())
sns.countplot(x=combined_df['Downfall'])
plt.title('Distribución de Clases: Caída vs No Caída')
plt.show()

# Escalar datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_df.drop(columns=['Downfall', 'Return']))

# Crear secuencias para LSTM
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, combined_df['Downfall'].values, seq_length)

# Aplicar SMOTE para balancear las clases
X_reshaped = X.reshape(X.shape[0], -1)  # Aplanar para SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
X_resampled = X_resampled.reshape(X_resampled.shape[0], seq_length, X.shape[2])

# Dividir datos balanceados en train y test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Construir un modelo LSTM optimizado
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Entrenar modelo
history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluar modelo
y_prob = model.predict(X_test)

# Búsqueda de umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
g_means = np.sqrt(tpr * (1-fpr))
optimal_idx = np.argmax(g_means)
optimal_threshold = thresholds[optimal_idx]

print(f"Umbral Óptimo: {optimal_threshold}")

# Predicción con umbral optimizado
y_pred = (y_prob > optimal_threshold).astype(int)

# Evaluar
rmse = np.sqrt(mean_squared_error(y_test, y_prob))
accuracy = accuracy_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Gráficos
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida por Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ROC Curve
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Caída', 'Caída'], yticklabels=['No Caída', 'Caída'])
plt.title('Matriz de Confusión LSTM')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Crear un rango de fechas para y_test
test_dates = combined_df.index[-len(y_test):]  # Últimas fechas según el tamaño de y_test

# Convertir y_test y las predicciones a pandas Series con fechas como índice
y_test_series = pd.Series(y_test, index=test_dates)
y_prob_series = pd.Series(y_prob.flatten(), index=test_dates)

# Seleccionar las fechas específicas
start_date = '2020-02-01'
end_date = '2020-05-31'
selected_data = y_test_series.loc[start_date:end_date]
selected_forecast_class = y_prob_series.loc[start_date:end_date]

# Graficar los valores reales y las predicciones
plt.figure(figsize=(12, 6))
plt.plot(selected_data, label='Real', color='blue')
plt.plot(selected_forecast_class, label='Predicho', color='red', linestyle='--')
plt.title('Caídas Reales vs Predichas (Febrero - Mayo 2020)')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Caída (1) / No Caída (0)')
plt.grid(True)
plt.show()