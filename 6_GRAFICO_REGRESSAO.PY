import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

# --- Caminho para os dados já tratados ---
caminho_arquivo = r"C:\Users\danie\OneDrive\Documentos\Mestrado-ITS-Article\ANÁLISE_FINAL\ANALISE_VOO_DETALHADA_v2_tempo_diferente.csv"
df = pd.read_csv(caminho_arquivo, low_memory=False)

# --- Reprocessamento rápido para o gráfico funcionar ---
# Converte colunas e gera variável PBN
implantation_years = {
    'SBCF': 2015, 'SBBR': 2015, 'SBCT': 2017, 'SBFZ': 0, 'SBPA': 2017, 'SBRF': 2010,
    'SBRJ': 2011, 'SBGL': 2011, 'SBSV': 2017, 'SBSP': 2015, 'SBGR': 2015
}
df['Ano'] = df['Ano'].astype(int)
df['PBN_orig'] = df.apply(lambda row: max(0, row['Ano'] - implantation_years.get(row['Sigla ICAO Aeroporto Origem'], 0)), axis=1)
df['PBN_dest'] = df.apply(lambda row: max(0, row['Ano'] - implantation_years.get(row['Sigla ICAO Aeroporto Destino'], 0)), axis=1)
df['PBN'] = (df['PBN_orig'] + df['PBN_dest']) / 2

# Dummies para modelo
df = pd.get_dummies(df, columns=['Modelo Equipamento'], drop_first=True)

# Seleção de variáveis independentes
indep_vars = ['PBN', 'Distância_Rota_NM']
dummy_cols = [col for col in df.columns if col.startswith('Modelo Equipamento_')]
X = df[indep_vars + dummy_cols]
X = sm.add_constant(X).astype(float)
y = df['Tempo_Voo_Real_min'].astype(float)

# --- Ajuste do modelo novamente (pode ser substituído por pickle no futuro) ---
modelo = sm.OLS(y, X).fit(cov_type='HC3')

# --- Gráfico 1: Resíduos vs Valores Ajustados ---
residuos = modelo.resid
ajustados = modelo.fittedvalues

plt.figure(figsize=(8, 5))
plt.scatter(ajustados, residuos, alpha=0.2, s=10)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores Ajustados")
plt.ylabel("Resíduos")
plt.title("Resíduos vs Ajustados (Tempo de Voo)")
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_residuos_vs_ajustados.png")
plt.close()

# --- Gráfico 2: Coeficientes da Regressão ---
coefs = modelo.params.drop('const')
errors = modelo.bse[coefs.index]

plt.figure(figsize=(10, 6))
coefs.plot(kind='bar', yerr=errors, capsize=4, color='skyblue')
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel("Coeficiente")
plt.title("Coeficientes da Regressão - Tempo de Voo")
plt.tight_layout()
plt.savefig("grafico_coeficientes_tempo.png")
plt.close()

# --- Gráfico 3: Efeito Marginal do PBN ---
pbn_vals = np.linspace(0, df['PBN'].max(), 100)
X_base = {
    'const': 1,
    'Distância_Rota_NM': df['Distância_Rota_NM'].mean()
}
for col in dummy_cols:
    X_base[col] = 0
X_base['Modelo Equipamento_B738'] = 1  # fixa como B738, por exemplo

y_pred_pbn = [modelo.predict({**X_base, 'PBN': val})[0] for val in pbn_vals]

plt.figure(figsize=(8, 5))
plt.plot(pbn_vals, y_pred_pbn, color='green')
plt.xlabel("Índice PBN")
plt.ylabel("Tempo de Voo Estimado (min)")
plt.title("Efeito Marginal Estimado do PBN")
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_efeito_pbn_tempo_voo.png")
plt.close()
