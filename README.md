import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, plotting , HRPOpt, black_litterman, objective_functions, BlackLittermanModel, DiscreteAllocation
import os
import ffn
from sklearn.neighbors import KernelDensity
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from pypfopt.risk_models import CovarianceShrinkage
from scipy.linalg import sqrtm
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid
import seaborn as sns
import os
import locale  
from tabulate import tabulate
from skopt import gp_minimize
from skopt.space import Real, Categorical
import scipy.stats as stats
import pyfolio as pf;

# Tickers de los 10 principales activos de los primeros 6 sectores del S&P 500

tickers = ['AAPL','NVDA','MSFT','AVGO','ORCL','CRM','PLTR','CSCO','ACN','IBM',
           'BRK-B','JPM','V','MA','BAC','WFC','GS','MS','AXP','C',
           'AMZN','TSLA','HD','MCD','BKNG','LOW','TJX','SBUX','NKE','MELI',
           'LLY','UNH','JNJ','ABBV','ABT','ISRG','MRK','TMO','AMGN','BSX',
           'GOOG','META','NFLX','TMUS','DIS','T','VZ','CMCSA','SPOT','DASH',
           'GE','CAT','RTX','UNP','BA','HON','DE','ETN','LMT','UPS']

# Parámetros
start_date = "2022-01-01"
end_date = "2025-01-01"
rf_annual = 0.0452  # Tasa libre de riesgo anual (4.52%)
num_clusters = 30  # Número de clústeres para K-Means
num_assets = 40   # Cantidad de activos a seleccionar

# Filtrar tickers con datos en el rango de fechas
valid_tickers = []

for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date,auto_adjust=False)
        if len(data) > 0:  # Verificar si hay datos
            valid_tickers.append(ticker)
    except Exception as e:
        print(f"Error al obtener datos para {ticker}: {e}")

print(f"Tickers con datos en el rango: {valid_tickers}")

#valido que esten todas
print(len(valid_tickers))

# Convertir la tasa libre de riesgo anual a mensual
rf_monthly = (1 + rf_annual) ** (1 / 12) - 1  
print(data)

# Descargar precios de cierre ajustados
data = yf.download(tickers, start=start_date, end=end_date,auto_adjust=False ,progress=False)["Adj Close"].dropna(axis=1, how='all')

# Calcular retornos diarios
returns = data.pct_change().dropna()


# Calcular el Ratio de Sharpe (sin anualizar aún)
sharpe_ratios = (returns.mean() - rf_annual / 252) / returns.std()

# Seleccionar las 40 empresas con mayor Ratio de Sharpe
top_40 = sharpe_ratios.nlargest(40).index.tolist()

# Descargar precios de los top 40 seleccionados
data = yf.download(top_40, start=start_date, end=end_date,auto_adjust=False)["Adj Close"]


# Calcular métricas
expected_returns = returns.mean() * 252  # Anualizar retornos promedio
volatility = returns.std() * np.sqrt(252)  # Anualizar volatilidad
sharpe_ratio = (expected_returns - rf_annual) / volatility  # Ratio de Sharpe anualizado

# Seleccionar los 40 activos con mejor Sharpe
top_assets = sharpe_ratio.nlargest(num_assets).index
filtered_returns = returns[top_assets]

# Clustering con K-Means
# Seleccionamos las métricas de retorno y volatilidad para los clusters
X = np.array([expected_returns[top_assets], volatility[top_assets]]).T

# Aplicamos K-Means para obtener los clústeres
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Crear un DataFrame con los resultados de los clusters
cluster_df = pd.DataFrame({
    'Asset': top_assets,
    'Expected Return': expected_returns[top_assets],
    'Volatility': volatility[top_assets],
    'Sharpe Ratio': sharpe_ratio[top_assets],
    'Cluster': clusters
})

# Ver los activos agrupados por clústeres
print(cluster_df.sort_values(by='Cluster'))


# Crear DataFrame con métricas para clustering
features = pd.DataFrame({'Return': expected_returns[top_assets], 'Volatility': volatility[top_assets]})

# Normalizar datos para K-Means
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Agregar los clusters al DataFrame
features['Cluster'] = clusters

# Aplicar clustering jerárquico para el dendrograma
linkage_matrix = linkage(features_scaled, method='ward')

# Seleccionar un activo por clúster (el más cercano al centroide)
selected_assets = []
for cluster in range(num_clusters):
    cluster_assets = features[features['Cluster'] == cluster]
    center = kmeans.cluster_centers_[cluster]
    distances = np.linalg.norm(scaler.transform(cluster_assets.iloc[:, :2]) - center, axis=1)
    selected_assets.append(cluster_assets.index[np.argmin(distances)])


# --- GRAFICAR CLUSTERS CON BURBUJAS Y CONEXIONES ---
plt.figure(figsize=(12, 8), dpi=300)  # Mayor resolución

# Definir colores para cada cluster
colors = plt.cm.Set2(np.linspace(0, 1, num_clusters))

# Gráfico de dispersión con burbujas
for cluster in range(num_clusters):
    cluster_assets = features[features['Cluster'] == cluster]
    
    # Puntos con burbujas
    plt.scatter(cluster_assets['Volatility'], cluster_assets['Return'], 
                s=cluster_assets['Volatility'] * 150,  # Ajuste de tamaño de burbujas
                alpha=0.75, label=f'Cluster {cluster+1}', edgecolors='k', 
                c=[colors[cluster]] * len(cluster_assets), linewidth=0.5)
    
    # Conexiones dentro del cluster (líneas entre activos)
    for i in range(len(cluster_assets)):
        for j in range(i+1, len(cluster_assets)):
            plt.plot([cluster_assets['Volatility'].iloc[i], cluster_assets['Volatility'].iloc[j]],
                     [cluster_assets['Return'].iloc[i], cluster_assets['Return'].iloc[j]], 
                     color='gray', alpha=0.25, linewidth=0.6, linestyle="dotted")

# Resaltar activos seleccionados
plt.scatter(features.loc[selected_assets]['Volatility'], features.loc[selected_assets]['Return'],
            color='red', marker='X', s=250, label='Seleccionados', edgecolors='k', linewidth=1.2)

# Etiquetas de los activos con pequeño desplazamiento
for i in range(len(features)):
    plt.text(features['Volatility'].iloc[i] + 0.001, features['Return'].iloc[i] + 0.001, 
             features.index[i], fontsize=8, alpha=0.85, color='black')

# Títulos y etiquetas
plt.title('Clustering de Activos con K-Means (Mejor Sharpe) ', fontsize=14, fontweight='bold')
plt.xlabel('Volatilidad', fontsize=12)
plt.ylabel('Rendimiento Esperado', fontsize=12)

# Optimizar leyenda para evitar superposiciones
legend = plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.05, 1), framealpha=0.8)
plt.grid(True, linestyle='--', alpha=0.5)

# Mostrar gráfico
plt.show()

# --- GRAFICAR DENDROGRAMA CON NOMBRES DE LOS ACTIVOS ---
plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, labels=features.index, leaf_rotation=90, leaf_font_size=10)
plt.title("Dendrograma de Clustering Jerárquico")
plt.xlabel("Activos")
plt.ylabel("Distancia")
plt.show()

# Mostrar activos seleccionados
print("Activos seleccionados por K-Means:", selected_assets)

data_assets = yf.download(selected_assets, start=start_date, end=end_date, progress=False,auto_adjust=False)["Adj Close"].dropna(axis=1, how='all')

# Calcular retornos diarios
returns_assets = data_assets.pct_change().dropna()
print(returns_assets)

# Calcular el Ratio de Sharpe (sin anualizar aún)
sharpe_ratios_assets = (returns_assets.mean() - rf_annual / 252) / returns_assets.std()

expected_returns_assets = returns_assets.mean()*252
print(expected_returns_assets)

# Crear un DataFrame con los resultados de los activos seleccionados
activos_seleccionados_df = pd.DataFrame({
    'Asset': selected_assets,
    'Expected Return': expected_returns[selected_assets],
    'Volatility': volatility[selected_assets],
    'Sharpe Ratio': sharpe_ratio[selected_assets]
})
# Ver los activos agrupados por clústeres
print(activos_seleccionados_df.sort_values(by='Expected Return'))

## Matriz Cov

plt.rcParams['figure.figsize'] = (10, 10)

cov_matrix_assets = returns_assets.cov()*252

# Guardar los nombres de los activos
asset_names_cov = cov_matrix_assets.index.tolist()

# Graficar manteniendo los nombres de los activos
plotting.plot_covariance(cov_matrix_assets, plot_correlation=True)
print(cov_matrix_assets)



##Graficar precios mensuales

# 📊 Crear figura más grande
plt.figure(figsize=(16, 8))

# 🔹 Graficar evolución de precios de los 30 activos
data_assets.plot(ax=plt.gca(), linewidth=1, alpha=0.8)

# 🎨 Ajustes visuales
plt.title('Evolución de Precios Mensuales', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precio', fontsize=12)

# 🔻 Mover la leyenda 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, ncol=1, frameon=False)

# 🛠 Ajustar layout para que todo se vea bien
plt.tight_layout()
plt.show()

##Definir risky assets

RISKY_ASSETS = selected_assets
n_assets = len(selected_assets)
N_MONTHS = 12

# Crear una figura con 5 filas y 6 columnas (30 subgráficos)
fig, axes = plt.subplots(5, 6, figsize=(30, 30))

# Iterar sobre los activos y sus respectivos subgráficos
for i, asset in enumerate(selected_assets):
    ax = axes[i // 6, i % 6]  # Calcular la posición del subgráfico
    asset_returns = returns_assets[asset]
    plot_title = f'{asset} returns: {start_date} - {end_date}'

    # Graficar los retornos
    asset_returns.plot(ax=ax, title=plot_title)
    
    # Configurar límites del eje X (fechas de 2020 a 2025)
    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    # Configurar el formato de fecha si es necesario
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))

# Ajustar el layout para evitar superposición de títulos
plt.tight_layout()
plt.show()


##CONSTRUCCIÓN DE PORTAFOLIOS

#### Market Cap. Cálculo de pesos iniciales
# Establecer localización para el formato monetario
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Función para obtener el market cap de los activos
def get_market_cap(RISKY_ASSETS):
    market_caps = {}
    for RISKY_ASSETS in RISKY_ASSETS:
        try:
            info = yf.Ticker(RISKY_ASSETS).info
            market_cap = info['marketCap']
            market_caps[RISKY_ASSETS] = locale.currency(market_cap, grouping=True)
        except:
            print(f"No se pudo obtener el market cap para {RISKY_ASSETS}")
    return market_caps

# Ejemplo de uso
market_caps = get_market_cap(RISKY_ASSETS)

# Convertir market cap a números
market_caps_numeric = {ticker: locale.atof(market_cap.strip('$')) for ticker, market_cap in market_caps.items()}

# Ordenar los activos por market cap de mayor a menor
sorted_assets = sorted(market_caps_numeric.keys(), key=lambda x: market_caps_numeric[x], reverse=True)

# Calcular los pesos iniciales del portafolio según el market cap
total_market_cap = sum(market_caps_numeric.values())
initial_weights = {asset: market_caps_numeric[asset] / total_market_cap for asset in sorted_assets}

# Crear DataFrame para la salida
output_data = []

# Llenar la lista de datos
for asset in sorted_assets:
    market_cap = market_caps[asset]
    weight = initial_weights[asset] * 100  # Convertir a porcentaje
    output_data.append({'Activo': asset, 'Market Cap ($)': market_cap, 'Participación (%)': weight})

# Crear DataFrame desde la lista de datos
output_df = pd.DataFrame(output_data)

##Convierto initial_weigths y retorno esperado en una lista (array)

initial_weights = np.array([initial_weights[asset] for asset in sorted_assets])
expected_returns_assets= np.array([expected_returns_assets[asset] for asset in sorted_assets])

##Veo como me queda el portafolio segun Market_Cap

# Rendimiento esperado del portafolio segun market cap
portfolio_return_marketcap = np.dot(initial_weights, expected_returns_assets)

# Volatilidad del portafolio segun market cap
portfolio_volatility_marketcap = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix_assets, initial_weights)))

# Ratio de Sharpe (usando una tasa libre de riesgo de 4.52% anual)
rf_rate = rf_annual
sharpe_ratio_marketcap = (portfolio_return_marketcap - rf_rate) / portfolio_volatility_marketcap

# Imprimir los pesos del portafolio
# Mostrar el DataFrame
print(output_df)

for asset, weight in zip(sorted_assets, initial_weights):
    print(f"Activo: {asset}, Peso: {weight:.2%}")
print(f"Rendimiento Esperado segun Market Cap: {portfolio_return_marketcap:.2%}")
print(f"Volatilidad segun Market Cap: {portfolio_volatility_marketcap:.2%}")
print(f"Ratio de Sharpe segun Market Cap: {sharpe_ratio_marketcap:.2f}")

### Modern Portfolio Theory (Markowitz) MAXIMIZAR SHARPE

# Función para optimizar Markowitz

def markowitz_optimizer(expected_returns_assets, cov_matrix_assets, initial_weights, rf_annual):
    num_assets = len(RISKY_ASSETS)
    
    # Restricción de suma de pesos = 1
    constraints = [{'type': 'eq', 'fun': lambda initial_weights: np.sum(initial_weights) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Función objetivo: maximizar Sharpe Ratio (minimizar -Sharpe para que sea una minimización)
    def negative_sharpe_ratio(initial_weights):
        portfolio_return_markowitz = np.dot(initial_weights, expected_returns_assets)
        portfolio_volatility_markowitz = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix_assets, initial_weights)))
        return -(portfolio_return_markowitz - rf_annual) / portfolio_volatility_markowitz  # Negativo para minimizar
    
    # Optimización con los pesos iniciales basados en market cap
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  # Retorna los pesos óptimos

# Optimización de Markowitz usando la data preprocesada
markowitz_weights = markowitz_optimizer(expected_returns_assets, cov_matrix_assets, initial_weights, rf_rate)

# Calcular rendimiento y volatilidad del portafolio optimizado
portfolio_return_markowitz = np.dot(markowitz_weights, expected_returns_assets)
portfolio_volatility_markowitz = np.sqrt(np.dot(markowitz_weights.T, np.dot(cov_matrix_assets, markowitz_weights)))
sharpe_ratio_max = (portfolio_return_markowitz - rf_rate) / portfolio_volatility_markowitz

# Convertir pesos a porcentajes
markowitz_weights_percent = markowitz_weights * 100
weights_df = pd.DataFrame({'Activo': sorted_assets, 'Peso (%)': markowitz_weights_percent.round(2)})

# Mostrar resultados
print("Pesos del Portafolio Máximo Ratio Sharpe:")
print(weights_df)
print(f"Rendimiento Esperado Máximo Ratio Sharpe: {portfolio_return_markowitz:.2%}")
print(f"Volatilidad Máximo Ratio Sharpe: {portfolio_volatility_markowitz:.2%}")
print(f"Ratio de Sharpe Máximo Ratio Sharpe: {sharpe_ratio_max:.2f}")

# Gráfica del portafolio optimizado
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatility_markowitz, portfolio_return_markowitz, marker='o', color='r', label='Portafolio Optimizado')

# Gráfica de los activos individuales
for i, asset in enumerate(sorted_assets):
    plt.scatter((cov_matrix_assets[i, i]), expected_returns_assets[i], marker='x', label=asset)

plt.title('Portafolio Optimizado vs Activos Individuales')
plt.xlabel('Volatilidad')
plt.ylabel('Rendimiento')
plt.legend()
plt.grid(True)
plt.show()


### Modern Portfolio Theory (Markowitz) MVO

# Función para optimizar minimizando la volatilidad (riesgo)
def min_volatility_optimizer(expected_returns_assets, cov_matrix_assets, initial_weights):
    num_assets = len(expected_returns_assets)

    # Restricción de suma de pesos = 1
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))  # Pesos entre 0% y 100%

    # Función objetivo: minimizar la volatilidad
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_assets, weights)))

    # Optimización
    result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x  # Retorna los pesos óptimos

# Optimización de Markowitz minimizando el riesgo
min_vol_weights = min_volatility_optimizer(expected_returns_assets, cov_matrix_assets, initial_weights)

# Calcular rendimiento y volatilidad del portafolio optimizado
portfolio_return_min_vol = np.dot(min_vol_weights, expected_returns_assets)
portfolio_volatility_min_vol = np.sqrt(np.dot(min_vol_weights.T, np.dot(cov_matrix_assets, min_vol_weights)))
sharpe_ratio_mvo = (portfolio_return_min_vol - rf_rate) / portfolio_volatility_min_vol

# Convertir pesos a porcentajes
min_vol_weights_percent = min_vol_weights * 100
weights_df = pd.DataFrame({'Activo': sorted_assets, 'Peso (%)': min_vol_weights_percent.round(2)})

# Mostrar resultados
print("Pesos del Portafolio minimizando la volatilidad:")
print(weights_df)
print(f"Rendimiento Esperado MVO: {portfolio_return_min_vol:.2%}")
print(f"Volatilidad MVO: {portfolio_volatility_min_vol:.2%}")
print(f"Ratio de Sharpe según Modern Portfolio Theory MVO: {sharpe_ratio_mvo:.2f}")



### Black Litterman Model

from scipy.optimize import minimize
import yfinance as yf
import numpy as np

def calculate_optimal_weights(expected_returns_assets, cov_matrix_assets):
    # Función de optimización para encontrar los pesos del portafolio
    objective_function = lambda initial_weights: -np.dot(initial_weights, expected_returns_assets)

    # Restricciones para los pesos (entre 0 y 1)
    constraints = ({'type': 'eq', 'fun': lambda initial_weights: np.sum(initial_weights) - 1})

    # Rango de pesos (entre 0 y 1)
    bounds = tuple((0, 1) for _ in range(len(expected_returns_assets)))

    # Optimización
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

from scipy.optimize import minimize

def calculate_portfolio_metrics(initial_weights, expected_returns_assets, cov_matrix_assets):
    portfolio_return_bl = np.dot(initial_weights, expected_returns_assets)
    portfolio_volatility_bl = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix_assets, initial_weights)))
    return portfolio_return_bl, portfolio_volatility_bl

def black_litterman(adj_close, tau, P, Q, RISKY_ASSETS):

    # Calcular el exceso de rendimiento esperado ajustado (pi)
    pi = tau * np.dot(cov_matrix_assets, expected_returns_assets)

    # Calcular la matriz Omega
    Omega = tau * np.dot(np.dot(P, cov_matrix_assets), P.T)

    # Función objetivo para la optimización
    objective_function = lambda initial_weights: np.linalg.norm(np.dot(P, initial_weights) - Q)

    # Restricciones para los pesos (positivos y que sumen 1)
    constraints = ({'type': 'eq', 'fun': lambda initial_weights: np.sum(initial_weights) - 1},
                   {'type': 'ineq', 'fun': lambda initial_weights: initial_weights})

    # Optimización
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=[(0, None)] * len(RISKY_ASSETS),
                      constraints=constraints)

    # Asignar nombres a los activos
    portfolio_dict = {RISKY_ASSETS[i]: result.x[i] for i in range(len(RISKY_ASSETS))}

    # Calcular rendimiento y volatilidad del portafolio
    portfolio_weights_bl = result.x
    portfolio_return_bl, portfolio_volatility_bl = calculate_portfolio_metrics(portfolio_weights_bl, expected_returns_assets, cov_matrix_assets)

    return portfolio_dict, portfolio_return_bl, portfolio_volatility_bl

# Ejemplo de uso

tau = 0.05
P = np.array([[-1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, -1, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, -1, 0, 0, 0,-0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, -1, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, -1, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 1,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,1,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,-1,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,-1,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,-1,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,-1,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,-1,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,-1,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,1,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,-1,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,-1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 1, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, -1, 0, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, -1, 0, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, -1, 0,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, -1,0,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,-1,0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,1,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,-1,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,-1,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,-1,0,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,-1,0,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,-1,0,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,-1,0],
              [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,-1],])
Q = np.array([[0.001], [0.141], [0.176], [0.072], [0.205], [0.186], [0.053], [0.197]
              , [0.710], [0.034], [0.076], [0.002], [0.168], [0.118], [0.042]
              , [0.009], [0.152], [0.060], [0.005], [0.107], [0.094], [0.012], [0.191]
              , [0.023], [0.106], [0.045], [0.196], [0.148], [0.056], [0.254]])
RISKY_ASSETS = asset_names_cov  # Asegúrate de que RISKY_ASSETS esté definido correctamente

portfolio_weights_bl, portfolio_return_bl, portfolio_volatility_bl = black_litterman(asset_names_cov, tau, P, Q, RISKY_ASSETS)
print("Pesos del Portafolio según Black-Litterman Model:")
for asset, weight in portfolio_weights_bl.items():
    print(asset, ":", "{:.2%}".format(weight))

# Convertir los pesos del portafolio a porcentaje
portfolio_weights_bl_percent = {asset: weight * 100 for asset, weight in portfolio_weights_bl.items()}


# Calcular el ratio de Sharpe para el portafolio Black-Litterman
def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_bl = rf_annual):
    return (portfolio_return - risk_free_bl) / portfolio_volatility

sharpe_ratio_bl = calculate_sharpe_ratio(portfolio_return_bl, portfolio_volatility_bl)

print(RISKY_ASSETS)
##Obtener resultados
print("Rendimiento del Portafolio de Black-Litterman:", "{:.2%}".format(portfolio_return_bl))
print("Volatilidad del Portafolio de Black-Litterman:", "{:.2%}".format(portfolio_volatility_bl))
print("Ratio de Sharpe del Portafolio de Black-Litterman:", "{:.2f}".format(sharpe_ratio_bl))

### HIERARCHICAL RISK PARITY

# Se optimiza el portafolio y se imprimen los resultados
hrp = HRPOpt(returns_assets)
hrp.optimize()
hrp_weights = hrp.clean_weights()
hrp_weights_percent = {asset: "{:.2f}%".format(initial_weights * 100) for asset, initial_weights in hrp_weights.items()}

# Imprimir los pesos del portafolio junto con los nombres de los activos
print("Pesos del Portafolio HRP:")
for asset, initial_weight_percent in hrp_weights_percent.items():
    print(f"{asset}: {initial_weight_percent}")

# Mostrar el rendimiento del portafolio
hrp.portfolio_performance(verbose=True);

#Agrego los weights de HRP al diccionario de portfolios
list_of_portfolios = {'HRP': dict(hrp_weights)}
portfolio_return_hrp, portfolio_volatility_hrp, sharpe_ratio_hrp = hrp.portfolio_performance()

# Plotear el dendrograma
plt.figure(figsize=(10, 6))
plotting.plot_dendrogram(hrp)
plt.title('Dendrograma de Hierarchical Risk Parity (HRP)')
plt.show()

# Crear un DataFrame con los pesos del portafolio HRP
df_weights = pd.DataFrame.from_dict(hrp_weights, orient='index', columns=['Peso'])

# Ordenar los activos por peso
df_weights = df_weights.sort_values(by='Peso', ascending=False)

# Crear el gráfico de mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(df_weights.T, cmap='YlGnBu', annot=True, fmt='.2%', cbar=False, linewidths=.5)
plt.title('Pesos del Portafolio HRP')
plt.xlabel('Activos')
plt.xticks(rotation=45)
plt.show()



# Optimización con restricciones VaR y CVaR

# Función para calcular VaR y CVaR


def VaR_CVaR(initial_weights, alpha=0.05):
    portfolio_returns_var = np.random.multivariate_normal(expected_returns_assets, cov_matrix_assets, 50000)
    portfolio_pnls_var = np.dot(portfolio_returns_var, initial_weights)
    VaR = np.percentile(portfolio_pnls_var, 100 * alpha)
    CVaR = portfolio_pnls_var[portfolio_pnls_var <= VaR].mean()
    return VaR, CVaR, portfolio_pnls_var

# Función objetivo: minimizar CVaR
def objective_function(weights):
    portfolio_return = np.dot(weights, expected_returns_assets)
    _, CVaR_value, _ = VaR_CVaR(weights)
    return - (portfolio_return - rf_annual) / abs(CVaR_value)  # Maximizar Sharpe ajustado por CVaR

    
# Restricciones y límites
num_assets = len(selected_assets)
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = tuple((0, 1) for _ in range(num_assets))


# Optimización
result = minimize(objective_function, initial_weights, method='trust-constr', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Cálculo de métricas finales
VaR_value, CVaR_value, portfolio_pnls = VaR_CVaR(optimal_weights)
portfolio_return_var = np.dot(optimal_weights, expected_returns_assets)
portfolio_volatility_var = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix_assets, optimal_weights)))
sharpe_ratio_var = (portfolio_return_var - rf_annual) / portfolio_volatility_var

# Mostrar resultados
print("Pesos Óptimos con restricciones de VaR y CVaR:")
for i, weight in enumerate(optimal_weights):
    print(f"{selected_assets[i]}: {weight*100:.2f}%")

print(f"VaR (5%): {VaR_value:.2%}")
print(f"CVaR (5%): {CVaR_value:.2%}")
print(f"Rendimiento Esperado: {portfolio_return_var:.2%}")
print(f"Volatilidad: {portfolio_volatility_var:.2%}")
print(f"Ratio de Sharpe: {sharpe_ratio_var:.2f}")

# Graficar distribución de rendimientos
plt.figure(figsize=(10, 6))
plt.hist(portfolio_pnls, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
plt.axvline(VaR_value, color='red', linestyle='dashed', linewidth=2, label=f'VaR (5%): {VaR_value:.2%}')
plt.axvline(CVaR_value, color='blue', linestyle='dashed', linewidth=2, label=f'CVaR (5%): {CVaR_value:.2%}')
plt.title('Distribución de Rendimientos del Portafolio con VaR y CVaR')
plt.xlabel('Rendimiento del Portafolio')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()


## Optimización Bayesiana
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real, Categorical

# Función para calcular VaR y CVaR (sin cambios)
def VaR_CVaR_bayesiana(initial_weights, alpha=0.05):
    portfolio_returns_var = np.random.multivariate_normal(expected_returns_assets, cov_matrix_assets, 50000)
    portfolio_pnls_var = np.dot(portfolio_returns_var, initial_weights)
    VaR = np.percentile(portfolio_pnls_var, 100 * alpha)
    CVaR = portfolio_pnls_var[portfolio_pnls_var <= VaR].mean()
    return VaR, CVaR, portfolio_pnls_var

# Función objetivo para la optimización bayesiana
def objective_function_bayesiana(weights):
    weights = np.array(weights)
    portfolio_return = np.dot(weights, expected_returns_assets)
    _, CVaR_value, _ = VaR_CVaR_bayesiana(weights)
    # Maximizar el Sharpe Ratio ajustado por CVaR
    return - (portfolio_return - rf_annual) / abs(CVaR_value)  # Minimización en lugar de maximización

# Definir el espacio de búsqueda para los pesos
space = [Real(0, 1) for _ in range(num_assets)]  # Espacio de búsqueda para los pesos

# Función objetivo: optimizar por el Sharpe ratio ajustado por CVaR
def objective_function_bayesiana(weights):
    weights = np.array(weights)
    # Normalizar los pesos para que sumen 1
    weights /= np.sum(weights)
    portfolio_return = np.dot(weights, expected_returns_assets)
    _, CVaR_value, _ = VaR_CVaR_bayesiana(weights)
    return - (portfolio_return - rf_annual) / abs(CVaR_value)  # Maximizar Sharpe ajustado por CVaR

# Realizar la optimización con gp_minimize
result_bayesiana = gp_minimize(objective_function_bayesiana, space, n_calls=100, random_state=42)

# Obtener los pesos óptimos después de la optimización
optimal_weights_bayesiana = np.array(result_bayesiana.x)
optimal_weights_bayesiana /= np.sum(optimal_weights_bayesiana)  # Asegurarse de que la suma sea 1

# Calcular las métricas finales con los pesos óptimos
VaR_value_bayesiana, CVaR_value_bayesiana, portfolio_pnls_bayesiana = VaR_CVaR_bayesiana(optimal_weights_bayesiana)
portfolio_return_bayesiana = np.dot(optimal_weights_bayesiana, expected_returns_assets)
portfolio_volatility_bayesiana = np.sqrt(np.dot(optimal_weights_bayesiana.T, np.dot(cov_matrix_assets, optimal_weights_bayesiana)))
sharpe_ratio_bayesiana = (portfolio_return_bayesiana - rf_annual) / portfolio_volatility_bayesiana

# Mostrar los resultados
print("Pesos Óptimos con Optimización Bayesiana:")
for i, weight in enumerate(optimal_weights_bayesiana):
    print(f"{selected_assets[i]}: {weight*100:.2f}%")

print(f"VaR (5%): {VaR_value_bayesiana:.2%}")
print(f"CVaR (5%): {CVaR_value_bayesiana:.2%}")
print(f"Rendimiento Esperado: {portfolio_return_bayesiana:.2%}")
print(f"Volatilidad: {portfolio_volatility_bayesiana:.2%}")
print(f"Ratio de Sharpe: {sharpe_ratio_bayesiana:.2f}")

# Graficar el histograma de los resultados de los PnL
plt.figure(figsize=(10,6))
plt.hist(portfolio_pnls_bayesiana, bins=100, color='blue', alpha=0.7, label='Distribución de PnLs')

# Marcar el VaR en el gráfico
plt.axvline(x=VaR_value_bayesiana, color='red', linestyle='--', label=f'VaR (5%): {VaR_value_bayesiana:.2%}')

# Marcar el CVaR en el gráfico
plt.axvline(x=CVaR_value_bayesiana, color='green', linestyle='--', label=f'CVaR (5%): {CVaR_value_bayesiana:.2%}')

# Añadir etiquetas y leyenda
plt.title('Histograma de PnLs del Portafolio con Optimización Bayesiana')
plt.xlabel('Pérdida y Ganancia (PnL)')
plt.ylabel('Frecuencia')

# Ajuste para mostrar las leyendas correctamente
plt.legend(loc='best', frameon=False)

# Mostrar el gráfico
plt.show()


# Simulación de Montecarlo para la Frontera Eficiente
def montecarlo_simulation(expected_returns_assets, num_portfolios=100000, rf_annual=rf_annual):
    
    results = np.zeros((3, num_portfolios))  # Matriz para almacenar rendimiento, riesgo y Sharpe
    weights_record = []
    
    for i in range(num_portfolios):
        weights_fe = np.random.random(len(RISKY_ASSETS))
        weights_fe /= np.sum(weights_fe)
        
        portfolio_return_fe = np.dot(weights_fe, expected_returns_assets)
        portfolio_volatility_fe = np.sqrt(np.dot(weights_fe.T, np.dot(cov_matrix_assets, weights_fe)))
        sharpe_ratio = (portfolio_return_fe - rf_annual) / portfolio_volatility_fe
        
        results[0, i] = portfolio_return_fe
        results[1, i] = portfolio_volatility_fe
        results[2, i] = sharpe_ratio
        weights_record.append(weights_fe)
    
    return results, weights_record

# Simulación de Montecarlo
num_simulations = 100000
montecarlo_results, _ = montecarlo_simulation(expected_returns_assets, num_simulations, rf_annual)

# Graficar la Frontera Eficiente
plt.figure(figsize=(10, 6))
plt.scatter(montecarlo_results[1, :], montecarlo_results[0, :], c=montecarlo_results[2, :], cmap='viridis', alpha=0.3)
plt.colorbar(label='Ratio de Sharpe')

# Ubicar cada portafolio en la gráfica
portfolios = {
    "Market Cap": (portfolio_volatility_marketcap, portfolio_return_marketcap),
    "Máx Sharpe": (portfolio_volatility_markowitz, portfolio_return_markowitz),
    "MVO ": (portfolio_volatility_min_vol, portfolio_return_min_vol),
    "Black-Litterman": (portfolio_volatility_bl, portfolio_return_bl),
    "HRP": (portfolio_volatility_hrp, portfolio_return_hrp),
    "Bayesiana": (portfolio_volatility_bayesiana, portfolio_return_bayesiana)
}

for name, (vol, ret) in portfolios.items():
    plt.scatter(vol, ret, marker='o', s=100, label=name)
    plt.text(vol, ret, name, fontsize=12, ha='right', va='bottom')

plt.xlabel('Volatilidad')
plt.ylabel('Rendimiento Esperado')
plt.title('Ubicación de Portafolios')
plt.legend()
plt.grid(True)
plt.show()
