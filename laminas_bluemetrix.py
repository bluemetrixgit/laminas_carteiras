import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CONFIGURAÇÃO INICIAL DO APP
# ============================================================

st.set_page_config(
    page_title="Lâmina Bluemetrix",
    page_icon="logo.png",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .stSelectbox, .stDateInput, .stDownloadButton {
        margin-bottom: 0.2rem;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .stDataFrame, .stImage {
        margin-top: -0.5rem;
    }

    .stApp {
        background-color: #111111;
        color: #f5f5f5;
    }

    table {
        background-color: #1e1e1e;
        color: #f5f5f5;
    }

    thead tr th {
        background-color: #2c2c2c;
        color: #f5f5f5;
    }

    tbody tr td {
        color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


st.title("📊 Lâmina Bluemetrix – Análise de Carteiras")


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:
    opcao_carteira = st.selectbox(
        "Selecione a carteira:",
        ["Carteira Cripto", "Carteira Internacional"]
    )

with col2:
    data_inicio = st.date_input(
        "Data de início:",
        value=dt.date(2022, 5, 1)
    )

    data_fim = st.date_input(
        "Data de fim:",
        value=dt.date.today()
    )


# ============================================================
# VALIDAÇÃO DAS DATAS
# ============================================================

if data_inicio >= data_fim:
    st.error("A data de início deve ser anterior à data de fim.")
    st.stop()


inicio = data_inicio.strftime("%Y-%m-%d")

# O yfinance utiliza a data final como exclusiva.
# Adicionamos 1 dia para incluir o último dia selecionado.
fim_download = (data_fim + dt.timedelta(days=1)).strftime("%Y-%m-%d")

valor_inicial = 100000


# ============================================================
# PESOS DAS CARTEIRAS
# ============================================================

pesos_cripto = {
    "BTC-USD": 0.60,
    "ETH-USD": 0.10,
    "SOL-USD": 0.08,
    "ONDO-USD": 0.03,
    "RENDER-USD": 0.02,
    "XRP-USD": 0.02,
    "IMX-USD": 0.02,
    "AAVE-USD": 0.02,
    "LINK-USD": 0.02,
    "LTC-USD": 0.02,
    "CASH": 0.07,
}

pesos_internacional = {
    "AMZN": 0.07,
    "BK": 0.05,
    "BRK-B": 0.80,
    "BKNG": 0.05,
    "CVX": 0.07,
    "GOOGL": 0.07,
    "IVV": 0.15,
    "JPM": 0.05,
    "JNJ": 0.05,
    "META": 0.05,
    "MSFT": 0.05,
    "NVDA": 0.03,
    "UBER": 0.07,
    "V": 0.05,
    "KO": 0.05,
    "CASH": 0.06,
}


pesos = (
    pesos_cripto
    if opcao_carteira == "Carteira Cripto"
    else pesos_internacional
)


benchmark_ticker = (
    "HASH11.SA"
    if opcao_carteira == "Carteira Cripto"
    else "^GSPC"
)


nome_benchmark = (
    "HASH11"
    if opcao_carteira == "Carteira Cripto"
    else "S&P500"
)


# ============================================================
# FUNÇÃO PARA BAIXAR DADOS DO YFINANCE
# ============================================================

def baixar_close(ticker, inicio, fim):
    """
    Baixa os preços de fechamento do ticker.
    Trata tanto o formato tradicional quanto o MultiIndex
    retornado por versões mais recentes do yfinance.
    """

    try:
        dados = yf.download(
            ticker,
            start=inicio,
            end=fim,
            auto_adjust=False,
            progress=False
        )

        if dados is None or dados.empty:
            return None

        # Caso tradicional:
        if "Close" in dados.columns:
            close = dados["Close"]

        # Caso MultiIndex:
        elif isinstance(dados.columns, pd.MultiIndex):
            if "Close" in dados.columns.get_level_values(0):
                close = dados["Close"]

                # Se vier como DataFrame com apenas um ticker
                if isinstance(close, pd.DataFrame):
                    if ticker in close.columns:
                        close = close[ticker]
                    else:
                        close = close.iloc[:, 0]
            else:
                return None

        else:
            return None

        # Garantir Series
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce")
        close = close.dropna()

        if close.empty:
            return None

        close.name = ticker

        return close

    except Exception as e:
        st.warning(f"Não foi possível obter os dados de {ticker}: {e}")
        return None


# ============================================================
# DOWNLOAD DOS ATIVOS
# ============================================================

precos = pd.DataFrame()

ativos_sem_dados = []


for ticker in pesos:

    if ticker == "CASH":
        continue

    serie = baixar_close(
        ticker,
        inicio,
        fim_download
    )

    if serie is not None:
        if precos.empty:
            precos = serie.to_frame()
        else:
            precos = precos.join(
                serie,
                how="outer"
            )
    else:
        ativos_sem_dados.append(ticker)


# ============================================================
# DOWNLOAD DO CASH
# ============================================================

if "CASH" in pesos:

    cdi = baixar_close(
        "IRFM11.SA",
        inicio,
        fim_download
    )

    if cdi is not None:
        cdi.name = "CASH"

        if precos.empty:
            precos = cdi.to_frame()
        else:
            precos = precos.join(
                cdi,
                how="outer"
            )
    else:
        ativos_sem_dados.append("CASH / IRFM11")


# ============================================================
# AVISO SOBRE ATIVOS SEM DADOS
# ============================================================

if ativos_sem_dados:

    st.warning(
        "Não foram encontrados dados para: "
        + ", ".join(ativos_sem_dados)
        + "."
    )


# ============================================================
# VALIDAÇÃO DOS PREÇOS
# ============================================================

if precos.empty:
    st.error(
        "Não foi possível obter dados de mercado para o período selecionado."
    )
    st.stop()


# ============================================================
# TRATAMENTO DOS PREÇOS
# ============================================================

precos.index = pd.to_datetime(
    precos.index,
    errors="coerce"
)

precos = precos.loc[
    ~precos.index.isna()
]

precos = precos.sort_index()


# Remove linhas em que algum ativo não possui preço.
precos = precos.dropna()


if precos.empty:
    st.error(
        "Após o tratamento dos dados, não existem datas em comum "
        "para todos os ativos da carteira."
    )
    st.stop()


# ============================================================
# RETORNOS DIÁRIOS
# ============================================================

retorno = precos.pct_change().dropna()


if retorno.empty:
    st.error(
        "Não foi possível calcular os retornos diários."
    )
    st.stop()


# ============================================================
# RETORNO DA CARTEIRA
# ============================================================

pesos_series = pd.Series(pesos)

# Garante que os pesos estejam na mesma ordem das colunas
pesos_series = pesos_series.reindex(
    retorno.columns
)

ret_carteira = (
    retorno * pesos_series
).sum(axis=1)


# ============================================================
# EVOLUÇÃO DA CARTEIRA
# ============================================================

carteira = (
    (1 + ret_carteira)
    .cumprod()
    * valor_inicial
)


# ============================================================
# BENCHMARK
# ============================================================

benchmark = baixar_close(
    benchmark_ticker,
    inicio,
    fim_download
)


if benchmark is None:

    st.error(
        f"Não foi possível obter os dados do benchmark {nome_benchmark}."
    )

    st.stop()


benchmark.index = pd.to_datetime(
    benchmark.index,
    errors="coerce"
)

benchmark = benchmark.loc[
    ~benchmark.index.isna()
]

benchmark = benchmark.sort_index()


benchmark_acum = (
    (1 + benchmark.pct_change())
    .cumprod()
    * valor_inicial
)


# ============================================================
# INTERSEÇÃO DAS DATAS
# ============================================================

idx_comum = carteira.index.intersection(
    benchmark_acum.index
)


if len(idx_comum) == 0:

    st.error(
        "Não existem datas em comum entre a carteira e o benchmark."
    )

    st.stop()


carteira = carteira.loc[idx_comum]

benchmark_acum = benchmark_acum.loc[idx_comum]


# ============================================================
# DATAFRAME FINAL
# ============================================================

df_final = pd.DataFrame({
    "Carteira": carteira.squeeze(),
    nome_benchmark: benchmark_acum.squeeze()
})


df_final = df_final.dropna()


# ============================================================
# GARANTIA DE DATAS CORRETAS
# ============================================================

df_final.index = pd.to_datetime(
    df_final.index,
    errors="coerce"
)

df_final = df_final.loc[
    ~df_final.index.isna()
]

df_final = df_final.sort_index()


# ============================================================
# VALIDAÇÃO FINAL
# ============================================================

if df_final.empty:

    st.error(
        "O DataFrame final está vazio. "
        "Não foi possível calcular a lâmina."
    )

    st.stop()


if len(df_final) < 2:

    st.error(
        "Não existem dados suficientes para calcular os indicadores."
    )

    st.stop()


# ============================================================
# RENTABILIDADE MENSAL
# ============================================================

df_mensal = df_final.resample("ME").last()

ret_mensal = (
    df_mensal
    .pct_change()
    .dropna()
    * 100
)


# ============================================================
# ÚLTIMOS 12 MESES
# ============================================================

ult_12m = ret_mensal.iloc[-12:].copy()


if not ult_12m.empty:

    ult_12m.index = pd.to_datetime(
        ult_12m.index
    ).strftime("%b/%y")


# ============================================================
# CÁLCULO DO YTD
# ============================================================

ano_atual = dt.datetime.today().year

inicio_ano = pd.Timestamp(
    f"{ano_atual}-01-01"
)


# Dados a partir do início do ano
df_ytd = df_final[
    df_final.index >= inicio_ano
].copy()


# ============================================================
# FUNÇÃO DE RENTABILIDADE
# ============================================================

def calcular_retorno_periodo(df, coluna):

    if df.empty:
        return np.nan

    serie = df[coluna].dropna()

    if len(serie) < 1:
        return np.nan

    valor_inicial_periodo = serie.iloc[0]
    valor_final_periodo = serie.iloc[-1]

    if valor_inicial_periodo == 0:
        return np.nan

    return (
        (valor_final_periodo / valor_inicial_periodo) - 1
    ) * 100


# ============================================================
# YTD
# ============================================================

retorno_ytd_carteira = calcular_retorno_periodo(
    df_ytd,
    "Carteira"
)

retorno_ytd_benchmark = calcular_retorno_periodo(
    df_ytd,
    nome_benchmark
)


# ============================================================
# 12 MESES
# ============================================================

if not ret_mensal.empty:

    retorno_12m_carteira = (
        ret_mensal["Carteira"]
        .iloc[-12:]
        .sum()
    )

    retorno_12m_benchmark = (
        ret_mensal[nome_benchmark]
        .iloc[-12:]
        .sum()
    )

else:

    retorno_12m_carteira = np.nan
    retorno_12m_benchmark = np.nan


# ============================================================
# RETORNO NO PERÍODO
# ============================================================

retorno_periodo_carteira = calcular_retorno_periodo(
    df_final,
    "Carteira"
)

retorno_periodo_benchmark = calcular_retorno_periodo(
    df_final,
    nome_benchmark
)


# ============================================================
# TABELA CONSOLIDADA
# ============================================================

consolidado = pd.DataFrame(
    index=[
        "Carteira",
        nome_benchmark
    ]
)


consolidado["Ano"] = [
    retorno_ytd_carteira,
    retorno_ytd_benchmark
]


consolidado["12 Meses"] = [
    retorno_12m_carteira,
    retorno_12m_benchmark
]


consolidado["No Período"] = [
    retorno_periodo_carteira,
    retorno_periodo_benchmark
]


# ============================================================
# ADICIONA MESES DOS ÚLTIMOS 12 MESES
# ============================================================

if not ult_12m.empty:

    for col in reversed(ult_12m.index):

        consolidado[col] = [
            ult_12m.loc[col, "Carteira"],
            ult_12m.loc[col, nome_benchmark]
        ]


tabela_lamina = consolidado.round(2)


# ============================================================
# RETORNOS DIÁRIOS
# ============================================================

ret_diario = (
    df_final
    .pct_change()
    .dropna()
)


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(series):

    if series.empty:
        return np.nan

    cum_max = series.cummax()

    drawdown = (
        (series - cum_max)
        / cum_max
    )

    return drawdown.min() * 100


# ============================================================
# INDICADORES
# ============================================================

indicadores = pd.DataFrame(
    index=[
        "Carteira",
        nome_benchmark
    ]
)


indicadores["Volatilidade Anual (%)"] = (
    ret_diario.std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# RETORNO ANUALIZADO
# ============================================================

numero_dias = len(df_final)


if numero_dias > 1:

    indicadores["Retorno Anual (%)"] = (
        (
            df_final.iloc[-1]
            / df_final.iloc[0]
        )
        ** (252 / numero_dias)
        - 1
    ) * 100

else:

    indicadores["Retorno Anual (%)"] = np.nan


# ============================================================
# SHARPE
# ============================================================

indicadores["Sharpe (CDI=9%)"] = (
    indicadores["Retorno Anual (%)"] - 9
) / indicadores["Volatilidade Anual (%)"]


# ============================================================
# MAX DRAWDOWN
# ============================================================

indicadores["Máx. Drawdown (%)"] = [
    max_drawdown(
        df_final["Carteira"]
    ),

    max_drawdown(
        df_final[nome_benchmark]
    )
]


indicadores = indicadores.round(2)


# ============================================================
# GRÁFICO
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 3)
)


fig.patch.set_facecolor("none")

ax.set_facecolor("none")


ax.plot(
    df_final.index,
    df_final["Carteira"],
    label="Carteira",
    color="#1f497d",
    linewidth=1.2
)


ax.plot(
    df_final.index,
    df_final[nome_benchmark],
    label=nome_benchmark,
    color="#a83232",
    linewidth=1.2
)


ax.yaxis.set_visible(False)

ax.set_ylabel("")

ax.set_xlabel("")


for spine in [
    "top",
    "right",
    "left",
    "bottom"
]:

    ax.spines[
        spine
    ].set_visible(False)


ax.grid(
    True,
    axis="x",
    linestyle="--",
    alpha=0.3,
    color="#ffffff"
)


ax.set_title(
    "Evolução Patrimonial",
    fontsize=10,
    color="white"
)


ax.tick_params(
    axis="x",
    colors="black"
)

ax.tick_params(
    axis="y",
    colors="black"
)


ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b/%y")
)


ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=9,
    frameon=False,
    labelcolor="black"
)


fig.tight_layout()


# ============================================================
# SALVAR GRÁFICO
# ============================================================

fig_path = "grafico_rentabilidade_bluemetrix.png"


fig.savefig(
    fig_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)


plt.close()


# ============================================================
# ESTILO DAS TABELAS
# ============================================================

def aplicar_estilo(df):

    return (
        df.style
        .set_properties(
            **{
                "background-color": "#1e1e1e",
                "color": "#f5f5f5",
                "border-color": "#444"
            }
        )
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("font-size", "10px"),
                    ("background-color", "#2c2c2c"),
                    ("color", "#f5f5f5")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("font-size", "10px"),
                    ("color", "#f5f5f5")
                ]
            }
        ])
    )


# ============================================================
# EXIBIÇÃO DO GRÁFICO
# ============================================================

st.markdown(
    "### 📈 Evolução Patrimonial"
)

st.image(
    fig_path,
    use_container_width=True
)


with open(fig_path, "rb") as f:

    st.download_button(
        "⬇️ Baixar Gráfico em PNG",
        f,
        "grafico_bluemetrix.png",
        "image/png"
    )


# ============================================================
# TABELA DE RENTABILIDADE
# ============================================================

st.markdown(
    "### 📊 Tabela de Rentabilidade Consolidada"
)

st.dataframe(
    aplicar_estilo(tabela_lamina),
    use_container_width=True
)


# ============================================================
# INDICADORES
# ============================================================

st.markdown(
    "### 📌 Indicadores Técnicos"
)

st.dataframe(
    aplicar_estilo(indicadores),
    use_container_width=True
)


# ============================================================
# INFORMAÇÕES DE DEBUG
# ============================================================

with st.expander("Informações dos dados"):

    st.write(
        f"Período solicitado: "
        f"{data_inicio.strftime('%d/%m/%Y')} "
        f"a {data_fim.strftime('%d/%m/%Y')}"
    )

    st.write(
        f"Primeira data disponível: "
        f"{df_final.index.min().strftime('%d/%m/%Y')}"
    )

    st.write(
        f"Última data disponível: "
        f"{df_final.index.max().strftime('%d/%m/%Y')}"
    )

    st.write(
        f"Quantidade de observações: "
        f"{len(df_final)}"
    )

    st.write(
        f"Quantidade de observações YTD: "
        f"{len(df_ytd)}"
    )

    st.write(
        f"Quantidade de observações mensais: "
        f"{len(df_mensal)}"
    )

    if ativos_sem_dados:

        st.write(
            "Ativos sem dados:",
            ativos_sem_dados
        )
