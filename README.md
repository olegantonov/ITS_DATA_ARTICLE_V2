# ITS_DATA_ARTICLE_V2

This repository contains the data processing scripts, regression models, and output files associated with the research article:


## 🔍 Research Summary

This study evaluates the impact of Performance-Based Navigation (PBN) on flight efficiency using data from over 350,000 domestic commercial flights in Brazil between 2010 and 2024. The analysis focuses on three key metrics:

- Actual Flight Time (minutes)  
- Fuel Consumption (kg)  
- CO₂ Emissions (kg)

We employ multiple linear regression models using PBN deployment levels, route distance, and aircraft type as explanatory variables. All scripts are written in Python using the `pandas`, `statsmodels`, `matplotlib`, and `seaborn` libraries.

## 📁 Repository Structure

```
ITS_DATA_ARTICLE_V2/
├── 00_DEPURAÇÃO.PY                      # Raw data cleaning and correction
├── 0_DOWNLOAD_DADOS.PY                 # Automated download of raw datasets
├── 1_CONSOLIDAÇÃO_DADOS.PY             # Concatenation and standardization of datasets
├── 2_ANÁLISE_DADOS_CONSOLIDADOS.PY     # Initial descriptive analysis
├── 3_FILTRO_AERÓDROMOS_VARIAVEIS.PY    # Filtering by airport and aircraft variables
├── 3.1_ANÁLISE_GRÁFICA_VOOS_POR_ANO.PY  # Graphical trend analysis by year
├── 4_ANÁLISE_E_INSERÇÃO_DADOS.PY       # Integration of derived variables (PBN index)
├── 4.1_FILTRO_TEMPOS_DIFERENTES.PY     # Removal of outliers in flight time
├── 4.3_ANÁLISE_VOOS.PY                  # General exploratory analysis
├── 5_1_ESTATÍSTICAS_DESCRITIVAS.PY     # Statistical summaries for all variables
├── 5_ANÁLISE_REGRESSÃO_LINEAR.PY       # Multiple linear regression (OLS)
└── 6_GRAFICO_REGRESSAO.PY              # Visualization of regression coefficients
```


## 📊 Methods

We developed and estimated three multiple linear regression models for the dependent variables:

1. **Actual Flight Time (min)**
2. **Fuel Consumption (kg)**
3. **CO₂ Emissions (kg)**

All models use:
- Heteroskedasticity-robust standard errors (HC3)
- Dummy variables for aircraft types (A320, A321, B737, B738)
- Route distance as a continuous predictor
- PBN implementation index as the key ITS-related independent variable

---

## 📁 Folders (to be created)

- `data/`: Raw and cleaned datasets (not public due to size/sensitivity)
- `results/`: Summary tables, regression outputs, and graphs
- `figures/`: Exported charts for publication

