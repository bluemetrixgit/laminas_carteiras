import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

# ========== CONFIGURAÇÃO INICIAL DO APP ==========
st.set_page_config(layout="wide")
st.markdown("""<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stSelectbox, .stDateInput {margin-bottom: -0.5rem;}
</style>""", unsafe_allow_html=True)
st.title("📊 Lâmina Bluemetrix – Análise de Carteiras (20/05/2025)")

# Input de seleção de carteira e data de início
col1, col2 = st.columns([1, 1])
with col1:
    opcao_carteira = st.selectbox("Selecione a carteira:", ["Carteira Cripto", "Carteira Internacional"])
with col2:
    data_inicio = st.date_input("Selecione a data de início:", value=dt.date(2022, 5, 1))

inicio = data_inicio.strftime("%Y-%m-%d")
fim = dt.datetime.today()
valor_inicial = 100000

# Pesos das carteiras
pesos_cripto = {
    "BTC-USD": 0.65,
    "ETH-USD": 0.10,
    "SOL-USD": 0.025,
    "LTC-USD": 0.025,
    "XRP-USD": 0.025,
    "ADA-USD": 0.05,
    "DOGE-USD": 0.025,
    "USDT-USD": 0.05,
    "CASH": 0.05,
}

pesos_internacional = {
    "AMZN": 0.10,
    "BK": 0.05,
    "BRK-B": 0.10,
    "CRM": 0.05,
    "CSCO": 0.05,
    "CVX": 0.05,
    "GOOGL": 0.10,
    "IVV": 0.15,
    "MSFT": 0.10,
    "NVDA": 0.08,
    "PYPL": 0.07,
    "SOXX": 0.10,
}

# Escolher carteira e benchmark
if opcao_carteira == "Carteira Cripto":
    pesos = pesos_cripto
    benchmark_ticker = "HASH11.SA"
else:
    pesos = pesos_internacional
    benchmark_ticker = "^GSPC"

# ===== DOWNLOAD DE DADOS =====
precos = pd.DataFrame()
for ticker in pesos:
    if ticker == "CASH":
        continue
    df = yf.download(ticker, start=inicio, end=fim)["Close"]
    df.name = ticker
    precos = precos.join(df, how="outer") if not precos.empty else df

# Preencher valor "CASH" como CDI aproximado usando IRFM11
if "CASH" in pesos:
    cdi = yf.download("IRFM11.SA", start=inicio, end=fim)["Close"]
    precos["CASH"] = cdi

# Benchmark
benchmark = yf.download(benchmark_ticker, start=inicio, end=fim)["Close"]

precos.dropna(inplace=True)
retorno = precos.pct_change().dropna()
ret_carteira = (retorno * pd.Series(pesos)).sum(axis=1)

carteira = (1 + ret_carteira).cumprod() * valor_inicial
benchmark_acum = (1 + benchmark.pct_change()).cumprod() * valor_inicial

idx_comum = carteira.index.intersection(benchmark_acum.index)
carteira = carteira.loc[idx_comum]
benchmark_acum = benchmark_acum.loc[idx_comum]

nome_benchmark = "HASH11" if opcao_carteira == "Carteira Cripto" else "S&P500"
df_final = pd.DataFrame({"Carteira": carteira.squeeze(), nome_benchmark: benchmark_acum.squeeze()}).dropna()

# ===== TABELA DE RENTABILIDADE =====
df_mensal = df_final.resample("M").last()
ret_mensal = df_mensal.pct_change().dropna() * 100
ult_12m = ret_mensal.iloc[-12:]
ult_12m.columns.name = None
ult_12m.index = ult_12m.index.strftime("%b/%Y")

ano_atual = dt.datetime.today().year
inicio_ano = f"{ano_atual}-01-01"
df_ytd = df_final[df_final.index >= inicio_ano]

consolidado = pd.DataFrame(index=["Carteira", nome_benchmark])
consolidado["Ano"] = [
    ((df_ytd["Carteira"].iloc[-1] / df_ytd["Carteira"].iloc[0]) - 1) * 100,
    ((df_ytd[nome_benchmark].iloc[-1] / df_ytd[nome_benchmark].iloc[0]) - 1) * 100
]
consolidado["12 Meses"] = [
    ret_mensal["Carteira"].iloc[-12:].sum(),
    ret_mensal[nome_benchmark].iloc[-12:].sum()
]
consolidado["No Período"] = [
    ((df_final["Carteira"].iloc[-1] / df_final["Carteira"].iloc[0]) - 1) * 100,
    ((df_final[nome_benchmark].iloc[-1] / df_final[nome_benchmark].iloc[0]) - 1) * 100
]
for col in reversed(ult_12m.index):
    consolidado[col] = [
        ult_12m.loc[col, "Carteira"],
        ult_12m.loc[col, nome_benchmark]
    ]
tabela_lamina = consolidado.round(2)

# ===== INDICADORES =====
ret_diario = df_final.pct_change().dropna()
def max_drawdown(series):
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min() * 100

indicadores = pd.DataFrame(index=["Carteira", nome_benchmark])
indicadores["Volatilidade Anual (%)"] = ret_diario.std() * np.sqrt(252) * 100
indicadores["Retorno Anual (%)"] = ((df_final.iloc[-1] / df_final.iloc[0]) ** (252 / len(df_final)) - 1) * 100
indicadores["Sharpe (CDI=9%)"] = (indicadores["Retorno Anual (%)"] - 9) / indicadores["Volatilidade Anual (%)"]
indicadores["Máx. Drawdown (%)"] = [max_drawdown(df_final["Carteira"]), max_drawdown(df_final[nome_benchmark])]
indicadores = indicadores.round(2)

# ===== GRÁFICO =====
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(df_final.index, df_final["Carteira"], label="Carteira", color="#1f4e79", linewidth=1.2)
ax.plot(df_final.index, df_final[nome_benchmark], label=nome_benchmark, color="#b30000", linewidth=1.2)
ax.yaxis.set_visible(False)
ax.set_ylabel("")
ax.set_xlabel("")
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.grid(True, axis='x', linestyle="--", alpha=0.4)
ax.set_title("Evolução Patrimonial", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=False)
fig.tight_layout()

# Salvar imagem e mostrar com botão de download
fig_path = "grafico_rentabilidade_bluemetrix.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
st.markdown("### 📈 Evolução Patrimonial")
st.image(fig_path, use_container_width=True)
with open(fig_path, "rb") as f:
    st.download_button(
        label="⬇️ Baixar Gráfico em PNG",
        data=f,
        file_name="grafico_bluemetrix.png",
        mime="image/png"
    )

# ===== EXIBIÇÃO DAS TABELAS =====
st.markdown("### 📊 Tabela de Rentabilidade Consolidada")
st.dataframe(tabela_lamina.style.set_table_styles([
    {'selector': 'th', 'props': [('font-size', '10px')]},
    {'selector': 'td', 'props': [('font-size', '10px')]}
]), use_container_width=True)

st.markdown("### 📌 Indicadores Técnicos")
st.dataframe(indicadores.style.set_table_styles([
    {'selector': 'th', 'props': [('font-size', '10px')]},
    {'selector': 'td', 'props': [('font-size', '10px')]}
]), use_container_width=True)
