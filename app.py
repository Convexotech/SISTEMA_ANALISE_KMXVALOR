"""Regressão linear simples (KM x Preço) com TensorFlow + Streamlit."""
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# --- Configuração -----------------------------------------------------------
SEED = 42
KM_MAX = 200_000
N_AMOSTRAS = 500
PRECO_BASE = 80_000.0          # preço de um carro com 0 km (R$)
DESVALORIZACAO_POR_KM = 0.30   # R$ perdidos por km rodado
RUIDO_STD = 4_000.0            # dispersão dos preços (R$)
FRACAO_TESTE = 0.20
EPOCAS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.05


@dataclass
class Artefatos:
    modelo: tf.keras.Model
    y_media: float
    y_desvio: float
    dados: pd.DataFrame
    r2: float
    mae: float


def gerar_dados() -> pd.DataFrame:
    """Dataset sintético: preço cai linearmente com o KM, mais ruído."""
    rng = np.random.default_rng(SEED)
    km = rng.uniform(0, KM_MAX, N_AMOSTRAS)
    preco = PRECO_BASE - DESVALORIZACAO_POR_KM * km + rng.normal(0, RUIDO_STD, N_AMOSTRAS)
    return pd.DataFrame({"km": km, "preco": np.clip(preco, 1_000, None)})


def prever(modelo: tf.keras.Model, y_media: float, y_desvio: float, km) -> np.ndarray:
    """Prediz o preço em R$ (desfaz a padronização do alvo)."""
    x = np.asarray(km, dtype="float32").reshape(-1, 1)
    z = modelo(x, training=False).numpy().ravel()
    return z * y_desvio + y_media


@st.cache_resource(show_spinner="Treinando o modelo...")
def treinar() -> Artefatos:
    tf.keras.utils.set_random_seed(SEED)
    dados = gerar_dados()

    # Split treino/teste com embaralhamento reprodutível
    idx = np.random.default_rng(SEED).permutation(len(dados))
    corte = int(len(dados) * (1 - FRACAO_TESTE))
    treino, teste = dados.iloc[idx[:corte]], dados.iloc[idx[corte:]]

    x_tr = treino[["km"]].to_numpy("float32")
    y_tr = treino["preco"].to_numpy("float32")
    y_media, y_desvio = float(y_tr.mean()), float(y_tr.std())

    # A normalização de x e y é o que faz o gradiente convergir rápido
    normalizacao = tf.keras.layers.Normalization(axis=-1)
    normalizacao.adapt(x_tr)

    modelo = tf.keras.Sequential(
        [tf.keras.Input(shape=(1,)), normalizacao, tf.keras.layers.Dense(1)]
    )
    modelo.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    modelo.fit(
        x_tr, (y_tr - y_media) / y_desvio,
        epochs=EPOCAS, batch_size=BATCH_SIZE, verbose=0,
    )

    # Avaliação no conjunto de teste, em R$
    y_te = teste["preco"].to_numpy()
    y_hat = prever(modelo, y_media, y_desvio, teste["km"].to_numpy())
    ss_res = float(np.sum((y_te - y_hat) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))

    return Artefatos(
        modelo=modelo, y_media=y_media, y_desvio=y_desvio, dados=dados,
        r2=1 - ss_res / ss_tot, mae=float(np.mean(np.abs(y_te - y_hat))),
    )


def brl(valor: float) -> str:
    return f"R$ {valor:,.0f}".replace(",", ".")


def grafico(art: Artefatos, km: int, preco: float) -> alt.Chart:
    grade = np.linspace(0, KM_MAX, 100)
    reta = pd.DataFrame({"km": grade, "preco": prever(art.modelo, art.y_media, art.y_desvio, grade)})
    selecionado = pd.DataFrame({"km": [km], "preco": [preco]})
    eixos = {"x": alt.X("km:Q", title="KM"), "y": alt.Y("preco:Q", title="Preço (R$)")}

    pontos = alt.Chart(art.dados).mark_circle(opacity=0.35, size=25).encode(**eixos)
    linha = alt.Chart(reta).mark_line(color="red", strokeWidth=3).encode(**eixos)
    alvo = alt.Chart(selecionado).mark_point(
        color="black", filled=True, size=150
    ).encode(**eixos)
    return (pontos + linha + alvo).properties(height=350)


def main() -> None:
    st.set_page_config(page_title="Preço por KM", page_icon="🚗")
    st.title("🚗 Previsão de preço por quilometragem")
    st.caption("Regressão linear treinada com TensorFlow em dados sintéticos.")

    art = treinar()

    km = st.slider("Quilometragem (KM)", 0, KM_MAX, 80_000, step=1_000)
    preco = max(0.0, float(prever(art.modelo, art.y_media, art.y_desvio, [km])[0]))

    col1, col2, col3 = st.columns(3)
    col1.metric("Preço estimado", brl(preco))
    col2.metric("R² (teste)", f"{art.r2:.3f}")
    col3.metric("Erro médio (MAE)", brl(art.mae))

    st.altair_chart(grafico(art, km, preco), use_container_width=True)

    with st.expander("Como funciona"):
        st.write(
            "O modelo é uma camada `Dense(1)` (y = w·x + b) precedida de normalização, "
            "treinada com MSE e Adam. Os dados são gerados no próprio app; "
            "para usar dados reais, substitua `gerar_dados()` por um `pd.read_csv(...)` "
            "com as colunas `km` e `preco`."
        )


main()
