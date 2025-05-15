import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import shapiro, normaltest

# Configuración inicial
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', 50)

# 1. Carga y preparación de datos
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

# Combinar todas las series temporales en un DataFrame unificado
combined_df = pd.DataFrame({
    'SP500': hist,
    'Volatilidad': vol,
    'Bonos 10 Años': dgs10_df['Precio_Bonos_10'],
    'Oro': o,
    'DXY': div,
    'WTI': wti_df['Precio_WTI'],
    'HGY': yield_,
})

# Rellenar fechas y valores faltantes
date_range = pd.date_range(start='2007-08-11', end='2025-01-17', freq='D')
combined_df.index = pd.to_datetime(combined_df.index)
combined_df = combined_df.reindex(date_range).ffill()

# Variables adicionales
# Variable objetivo (drawdown > 1%)
combined_df['Drawdown'] = (combined_df['SP500'].pct_change().shift(-1) < -0.01).astype(int)

# Retornos y métricas de momentum
combined_df['Return_1d'] = combined_df['SP500'].pct_change()
combined_df['Return_5d'] = combined_df['SP500'].pct_change(5)
combined_df['Return_21d'] = combined_df['SP500'].pct_change(21)
combined_df['MA_7'] = combined_df['SP500'].rolling(window=7).mean()
combined_df['MA_21'] = combined_df['SP500'].rolling(window=21).mean()
combined_df['MA_200'] = combined_df['SP500'].rolling(window=200).mean()
combined_df['MA_7_21_Ratio'] = combined_df['MA_7'] / combined_df['MA_21']
combined_df['Volatilidad_21d'] = combined_df['Return_1d'].rolling(window=21).std()

# Ratios entre activos
combined_df['Oro_SP500_Ratio'] = combined_df['Oro'] / combined_df['SP500']
combined_df['WTI_SP500_Ratio'] = combined_df['WTI'] / combined_df['SP500']

# Indicadores técnicos RSI y MACD
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

combined_df['RSI_14'] = calculate_rsi(combined_df['SP500'])
combined_df['MACD'] = combined_df['SP500'].ewm(span=12).mean() - combined_df['SP500'].ewm(span=26).mean()

# Lags de variables importantes
for lag in [1, 2, 3, 5, 10]:
    combined_df[f'Volatilidad_lag_{lag}'] = combined_df['Volatilidad'].shift(lag)
    combined_df[f'Drawdown_lag_{lag}'] = combined_df['Drawdown'].shift(lag)

# Rellenar valores faltantes
combined_df = combined_df.ffill().bfill()

# 2. Análisis de multicolinealidad
# Calcular VIF para las variables
vif_data = pd.DataFrame()
vif_data["Variable"] = combined_df.columns.drop(['Drawdown', 'SP500'])
vif_data["VIF"] = [variance_inflation_factor(combined_df.drop(['Drawdown', 'SP500'], axis=1).values, i)
                    for i in range(len(combined_df.columns.drop(['Drawdown', 'SP500'])))]

print("VIF Analysis:")
print(vif_data.sort_values('VIF', ascending=False).head(15))

# Eliminar variables con alta multicolinealidad (VIF > 10)
high_vif = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
combined_df = combined_df.drop(columns=high_vif)

# 3. Preparación de datos para RF
X = combined_df.drop(columns=['Drawdown', 'SP500'])
y = combined_df['Drawdown']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. División de datos con validación temporal
train_size = int(0.7 * len(X))
X_train, X_validacion = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_validacion = y[:train_size], y[train_size:]

# 5. Balanceo de clases, calcular class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
for clase, peso in class_weight_dict.items():
    print(f"Peso por clase {clase} = {peso}")

# 6. Modelo Random Forest con Calibración de Probabilidades
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
# Entrenar el modelo
# Calibración isotónica para mejorar probabilidades
calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=3)
calibrated_rf.fit(X_train, y_train)

# 7. Evaluación
y_prob = calibrated_rf.predict_proba(X_validacion)[:, 1]

# 8. Optimización de umbral con índice de Youden
fpr, tpr, thresholds = roc_curve(y_validacion, y_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"\nUmbral óptimo: {optimal_threshold:.3f}")

# Predicciones con umbral optimizado
y_pred = (y_prob > optimal_threshold).astype(int)

# 10. Evaluar
rmse = np.sqrt(mean_squared_error(y_validacion, y_prob))
accuracy = accuracy_score(y_validacion, y_pred)
print(f"RMSE: {rmse}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_validacion, y_pred))
print("ROC AUC Score:", roc_auc_score(y_validacion, y_prob))

# 11.Curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve RF 2007')
plt.show()

# 10. Matriz de confusión
conf_matrix = confusion_matrix(y_validacion, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Caída', 'Caída'], yticklabels=['No Caída', 'Caída'])
plt.title('Matriz de Confusión RF 2007')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# 11.Feature Importance (Gini)
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Features Random Forest 2007')
plt.show()

# 12. Backtest temporal
# Crear DataFrame con predicciones
results_df = combined_df.iloc[train_size:].copy()
results_df['Predicted_Probability'] = y_prob
results_df['Predicted_Class'] = y_pred

# Gráfico de probabilidades predichas vs real
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(results_df.index, results_df['Predicted_Probability'], label='Predicted Probability', alpha=0.7)
plt.scatter(results_df.index, results_df['Drawdown'], label='Actual Drawdown', color='red', alpha=0.3)
plt.title('Caídas Reales vs Predichas RF 2007')
plt.ylabel('Probability / Actual')
plt.legend()
plt.grid(True)

# 13. Crear un rango de fechas
test_dates = combined_df.index[-len(y_validacion):]

# Convertir las predicciones con fechas como índice
y_test_series = pd.Series(y_validacion, index=test_dates)
y_prob_series = pd.Series(y_prob.flatten(), index=test_dates)

# Seleccionar las fechas específicas
start_date = '2020-02-01'
end_date = '2020-05-31'
selected_data = y_test_series.loc[start_date:end_date]
selected_forecast_class = y_prob_series.loc[start_date:end_date]

# Graficar los valores reales y las predicciones
plt.subplot(1, 2, 2)
plt.plot(selected_data, label='Real', color='blue')
plt.plot(selected_forecast_class, label='Predicho', color='red', linestyle='--')
plt.title('Caídas Reales vs Predichas (Febrero - Mayo 2020) RF 2007')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Caída (1) / No Caída (0)')
plt.grid(True)
plt.show()

# 14. Análisis de residuos del modelo
residuals = y_validacion - y_prob

# ACF de los residuos
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plot_acf(residuals, lags=30, ax=plt.gca())
plt.title("ACF de los residuos del modelo RF 2007")
plt.tight_layout()

# PACF de los residuos
plt.subplot(1, 3, 2)
plot_pacf(residuals, lags=30, method='ywm', ax=plt.gca())
plt.title("PACF de los residuos del modelo RF 2007")
plt.tight_layout()

plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.title("Distribución de Residuos RF 2007")
plt.tight_layout()
plt.show()

# Tests estadísticos
shapiro_test = shapiro(residuals)
dagostino_test = normaltest(residuals)
print("\nTests de Normalidad:")

print(f"Shapiro-Wilk Test:\n  Estadístico = {shapiro_test.statistic:.4f}, p-valor = {shapiro_test.pvalue:.2e}")
print(f"D'Agostino Test:\n  Estadístico = {dagostino_test.statistic:.4f}, p-valor = {dagostino_test.pvalue:.2e}")

# Test de Ljung-Box para autocorrelación significativa
ljung_box = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"Test de Ljung-Box (lag 10):\n  Estadístico = {ljung_box['lb_stat'].values[0]:.2f}, p-valor = {ljung_box['lb_pvalue'].values[0]:.4f}")

# 15. Función para calcular métricas durante el entrenamiento
def train_with_metrics(model, X_train, y_train, X_validacion, y_validacion, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_auc, val_auc = [], []

    for train_index, test_index in tscv.split(X_train):
        X_tr, X_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_tr, y_tr)

        # ROC AUC en train
        y_train_prob = model.predict_proba(X_tr)[:, 1]
        train_auc.append(roc_auc_score(y_tr, y_train_prob))

        # ROC AUC en validation
        y_val_prob = model.predict_proba(X_te)[:, 1]
        val_auc.append(roc_auc_score(y_te, y_val_prob))

    return train_auc, val_auc


# Entrenar y obtener métricas
train_auc, val_auc = train_with_metrics(rf_model, X_train, y_train, X_validacion, y_validacion)

# Gráfico de entrenamiento vs validación
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_auc) + 1), train_auc, label='Train ROC AUC', marker='o')
plt.plot(range(1, len(val_auc) + 1), val_auc, label='Validation ROC AUC', marker='o')
plt.xlabel('Fold')
plt.ylabel('ROC AUC Score')
plt.title('Training vs Validation ROC AUC RF 2007')
plt.legend()
plt.grid(True)
plt.show()

# 16. Feature Importance
def permutation_feature_importance(model, X, y, feature_names, n_repeats=10):
    baseline_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
    importance_scores = {}

    n_features = X.shape[1]
    feature_names = feature_names[:n_features]

    for i, name in zip(range(n_features), feature_names):
        X_perturbed = X.copy()
        for _ in range(n_repeats):
            np.random.shuffle(X_perturbed[:, i])
            score = roc_auc_score(y, model.predict_proba(X_perturbed)[:, 1])
            importance_scores[name] = importance_scores.get(name, 0) + (baseline_score - score)

        importance_scores[name] /= n_repeats

    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': [importance_scores[name] for name in feature_names]
    }).sort_values('Importance', ascending=False)
# Uso con Random Forest
feature_importance = permutation_feature_importance(
    calibrated_rf,
    X_validacion,
    y_validacion,
    combined_df.columns.drop('Drawdown')
)

# Gráfico
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Permutation Feature Importance - Random Forest 2007')
plt.tight_layout()
plt.show()