import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. Leitura dos dados
caminho_arquivo = r"C:\Users\danie\OneDrive\Documentos\Mestrado-ITS-Article\ANÁLISE_FINAL\ANALISE_VOO_DETALHADA_v3_tempo_diferente.csv"
df = pd.read_csv(caminho_arquivo, low_memory=False)

# 2. Conversão das colunas numéricas e imputação de valores faltantes com a mediana
colunas_numericas = ['Ano', 'Distância_Rota_NM', 'Load_Factor_Utilizado', 'Número de Assentos',
                     'Tempo_Voo_Real_min', 'Consumo_Combustível_kg', 'Consumo_CO2_do_Voo_kg']
for col in colunas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 3. Dados de implantação do PBN
implantation_dates = {
    'SBCF': '12/11/2015', 'SBBR': '12/11/2015', 'SBCT': '12/10/2017',
    'SBFZ': None, 'SBPA': '12/10/2017', 'SBRF': '08/04/2010',
    'SBRJ': '28/07/2011', 'SBGL': '28/07/2011', 'SBSV': '27/04/2017',
    'SBSP': '12/11/2015', 'SBGR': '12/11/2015'
}
implantation_years = {}
for icao, date_str in implantation_dates.items():
    if date_str is None or str(date_str).strip() == '':
        implantation_years[icao] = 0
    else:
        dt = datetime.strptime(date_str, '%d/%m/%Y')
        implantation_years[icao] = dt.year

df['Ano'] = df['Ano'].astype(int)

# 4. Cálculo da variável PBN
def compute_PBNTMA(ano_voo, ano_implantacao):
    if ano_implantacao == 0:
        return 0
    if ano_voo >= ano_implantacao:
        return min(ano_voo - ano_implantacao, 2024 - ano_implantacao)
    return 0

df['PBN_orig'] = df.apply(lambda row: compute_PBNTMA(row['Ano'], implantation_years.get(row['Sigla ICAO Aeroporto Origem'], 0)), axis=1)
df['PBN_dest'] = df.apply(lambda row: compute_PBNTMA(row['Ano'], implantation_years.get(row['Sigla ICAO Aeroporto Destino'], 0)), axis=1)
df['PBN'] = (df['PBN_orig'] + df['PBN_dest']) / 2

# 5. Dummies para modelo de equipamento
df = pd.get_dummies(df, columns=['Modelo Equipamento'], drop_first=True)

# 6. Definição das variáveis independentes
indep_vars = ['PBN', 'Distância_Rota_NM']
dummy_cols = [col for col in df.columns if col.startswith('Modelo Equipamento_')]
X_base = df[indep_vars + dummy_cols].copy()
X_base = X_base.apply(pd.to_numeric, errors='coerce').fillna(X_base.median())

# 7. Função para ajuste de modelo e exportação de resultados
def ajustar_modelo(y, X, outcome_label):
    from io import StringIO
    import sys

    X_model = sm.add_constant(X).astype(float)
    y = y.astype(float)
    modelo = sm.OLS(y, X_model).fit(cov_type='HC3')

# Redirecionar saída
    buffer = StringIO()
    sys.stdout = buffer

    print(f"\n=== Modelo para {outcome_label} ===")
    with pd.option_context('display.float_format', '{:.8f}'.format):
        print(modelo.summary())

    # Teste de Breusch-Pagan
    bp_test = het_breuschpagan(modelo.resid, X_model)
    bp_labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print("\nTeste de Breusch-Pagan:")
    for label, value in zip(bp_labels, bp_test):
        print(f"{label}: {value:.4f}")

    # Observações influentes
    influence = modelo.get_influence()
    cooks = influence.cooks_distance[0]
    high_influence = np.where(cooks > 4 / len(X_model))[0]
    if len(high_influence) > 0:
        print("\nObservações com alta influência (Cook's distance > 4/n):")
        print(high_influence)
    else:
        print("\nNenhuma observação com alta influência identificada.")

    # Normalidade dos resíduos
    residuos = modelo.resid
    dagostino_stat, dagostino_pvalor = st.normaltest(residuos)
    print(f"\nTeste de D'Agostino & Pearson para {outcome_label}:")
    print(f"Estatística: {dagostino_stat:.4f}, p-valor: {dagostino_pvalor:.6f}")
    if dagostino_pvalor < 0.05:
        print("Conclusão: Resíduos NÃO seguem distribuição normal.")
    else:
        print("Conclusão: Não rejeitamos hipótese de normalidade.")

    # Efeito da variável PBN
    coef_pbn = modelo.params.get('PBN', float('nan'))
    pvalor_pbn = modelo.pvalues.get('PBN', float('nan'))
    print(f"\nEfeito da variável PBN em {outcome_label}:")
    print(f"Coeficiente: {coef_pbn:.6f}, p-valor: {pvalor_pbn:.8f}")
    if coef_pbn < 0 and pvalor_pbn < 0.05:
        print("Interpretação: Maior implantação de PBN está associada a uma redução significativa em", outcome_label)
    elif coef_pbn > 0 and pvalor_pbn < 0.05:
        print("Interpretação: Maior implantação de PBN está associada a um aumento significativo em", outcome_label)
    else:
        print("Interpretação: O efeito da variável PBN não é estatisticamente significativo.")

    # Restaurar saída padrão
    sys.stdout = sys.__stdout__
    nome_arquivo = f"saida_modelo_{outcome_label.replace(' ', '_').replace('(', '').replace(')', '')}.txt"
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())

    # Gráficos
    plt.figure(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color='blue')
    plt.title(f"Histograma dos Resíduos - {outcome_label}")
    plt.xlabel("Resíduos")
    plt.ylabel("Frequência")
    plt.savefig(f"hist_residuos_{outcome_label}.png")
    plt.close()

    sm.qqplot(residuos, line='45', fit=True)
    plt.title(f"QQ-Plot dos Resíduos - {outcome_label}")
    plt.savefig(f"qqplot_residuos_{outcome_label}.png")
    plt.close()

    return modelo

# 8. Ajuste dos modelos
y_tempo = df['Tempo_Voo_Real_min'].fillna(df['Tempo_Voo_Real_min'].median())
modelo_tempo = ajustar_modelo(y_tempo, X_base, "Tempo de Voo Real (min)")

y_combustivel = df['Consumo_Combustível_kg'].fillna(df['Consumo_Combustível_kg'].median())
modelo_combustivel = ajustar_modelo(y_combustivel, X_base, "Consumo de Combustível (kg)")

y_co2 = df['Consumo_CO2_do_Voo_kg'].fillna(df['Consumo_CO2_do_Voo_kg'].median())
modelo_co2 = ajustar_modelo(y_co2, X_base, "Emissão de CO2 do Voo (kg)")
