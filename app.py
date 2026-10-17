"""Local portfolio analyzer and optimizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from portfolio import PERIODS, optimize, portfolio_metrics, read_prices, validate_allocations


st.set_page_config(page_title="Анализ портфеля", page_icon="📈", layout="wide")
st.title("Анализ и оптимизация портфеля")
st.caption("Исторические цены → показатели риска → варианты распределения капитала")

with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader("Цены активов, CSV", type="csv")
    st.caption("Формат: Date, затем столбец цен для каждого актива. Цены должны быть скорректированы на сплиты и дивиденды, если это важно для анализа.")
    if uploaded is None:
        st.download_button("Скачать пример CSV", data=(Path(__file__).parent / "data/example_prices.csv").read_bytes(),
                           file_name="example_prices.csv", mime="text/csv")
    frequency = st.selectbox("Частота цен", list(PERIODS))
    annual_rf_pct = st.number_input("Безрисковая ставка, % годовых", value=0.0,
                                    min_value=-50.0, max_value=100.0, step=0.5)

if uploaded is None:
    st.info("Загрузите CSV с историческими ценами. Для знакомства скачайте пример слева.")
    st.stop()

try:
    prices = read_prices(uploaded)
except Exception as exc:
    st.error(str(exc))
    st.stop()

selected = st.multiselect("Активы", prices.columns.tolist(), default=prices.columns.tolist())
if len(selected) < 2:
    st.warning("Выберите минимум два актива.")
    st.stop()
prices = prices[selected]
returns = prices.pct_change().iloc[1:]
periods = PERIODS[frequency]
st.caption(f"Период: {prices.index.min():%d.%m.%Y} — {prices.index.max():%d.%m.%Y}; "
           f"{len(returns)} интервалов доходности. Годовых периодов: {periods}.")

default = pd.DataFrame({
    "Актив": selected,
    "Текущая доля, %": np.full(len(selected), 100 / len(selected)),
    "Мин., %": np.zeros(len(selected)),
    "Макс., %": np.full(len(selected), 100.0),
})
st.subheader("Текущий портфель и ограничения")
st.caption("Укажите текущие доли и допустимый диапазон для каждого актива. Все значения — в процентах.")
table = st.data_editor(default, hide_index=True, use_container_width=True,
                       disabled=["Актив"], num_rows="fixed",
                       column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=100.0,
                                                                          step=0.1, format="%.2f")
                                      for c in default.columns if c != "Актив"})

if st.button("Рассчитать", type="primary"):
    try:
        current, lower, upper = validate_allocations(table, selected)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    results = {"Текущий": current}
    failures = []
    with st.spinner("Расчёт оптимальных долей..."):
        for name, value in optimize(returns, lower, upper, annual_rf_pct / 100, periods).items():
            if isinstance(value, str):
                failures.append(f"{name}: {value}")
            else:
                results[name] = value
    for failure in failures:
        st.warning(f"Не удалось рассчитать {failure}")
    st.subheader("Сравнение портфелей")
    metrics = pd.DataFrame({name: portfolio_metrics(returns, weights, periods)
                            for name, weights in results.items()}).T
    st.dataframe(metrics.style.format("{:.2%}"), use_container_width=True)
    st.caption("Доходность — историческое среднее × число периодов в году; волатильность — стандартное отклонение × √число периодов. VaR и CVaR показывают потери за один период на уровне 95%; максимальная просадка рассчитана по истории при восстановлении целевых долей каждый период.")
    weights_df = pd.DataFrame({name: weights * 100 for name, weights in results.items()}, index=selected)
    st.subheader("Доли активов, %")
    st.dataframe(weights_df.style.format("{:.2f}%"), use_container_width=True)
    st.bar_chart(weights_df)
    st.download_button("Скачать доли CSV", weights_df.rename_axis("Asset").to_csv().encode("utf-8-sig"),
                       "portfolio_weights.csv", "text/csv")
    st.caption("Оптимизация использует исторические данные, без прогноза будущих доходностей и без учёта комиссий и налогов.")
