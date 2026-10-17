"""Validation, historical risk metrics, and Riskfolio-Lib optimization."""

from __future__ import annotations

import csv
from io import BytesIO

import numpy as np
import pandas as pd


PERIODS = {"Дневные": 252, "Недельные": 52, "Месячные": 12}


def read_prices(file: BytesIO | str) -> pd.DataFrame:
    """Read wide CSV: Date, ASSET_1, ASSET_2, ..."""
    if hasattr(file, "seek"):
        file.seek(0)
        first_line = file.readline()
        file.seek(0)
    else:
        with open(file, "rb") as stream:
            first_line = stream.readline()
    if isinstance(first_line, bytes):
        first_line = first_line.decode("utf-8-sig")
    header = next(csv.reader([first_line]))
    if len(header) != len(set(header)):
        raise ValueError("Названия столбцов должны быть уникальными.")
    raw = pd.read_csv(file)
    if raw.shape[1] < 3:
        raise ValueError("Нужны столбец Date и минимум два актива.")
    if raw.columns[0].strip().lower() != "date":
        raise ValueError("Первый столбец должен называться Date.")
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    if raw["Date"].isna().any() or raw["Date"].duplicated().any():
        raise ValueError("Проверьте даты: пустые, неверные и повторяющиеся даты недопустимы.")
    raw = raw.sort_values("Date").set_index("Date")
    try:
        prices = raw.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Все цены должны быть числами.") from exc
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Цены должны быть положительными, без пропусков и бесконечностей.")
    if len(prices) < 31:
        raise ValueError("Нужно минимум 31 наблюдение цен (30 периодов доходности).")
    return prices


def validate_allocations(table: pd.DataFrame, assets: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if table["Актив"].tolist() != assets:
        raise ValueError("Список активов в таблице отличается от выбранных данных.")
    try:
        values = table[["Текущая доля, %", "Мин., %", "Макс., %"]].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Все доли и границы должны быть числами.") from exc
    if not np.isfinite(values).all():
        raise ValueError("В долях и границах не должно быть пропусков.")
    current, lower, upper = (values[:, i] / 100 for i in range(3))
    if np.any(current < 0) or not np.isclose(current.sum(), 1, atol=1e-5):
        raise ValueError("Текущие доли должны быть неотрицательными и в сумме давать 100%.")
    if np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper):
        raise ValueError("Границы должны удовлетворять 0 ≤ минимум ≤ максимум ≤ 100%.")
    if lower.sum() > 1 + 1e-8 or upper.sum() < 1 - 1e-8:
        raise ValueError("Ограничения несовместимы: сумма минимумов > 100% или максимумов < 100%.")
    return current, lower, upper


def portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray, periods: int) -> dict[str, float]:
    series = returns.to_numpy() @ weights
    wealth = np.cumprod(1 + series)
    drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1
    loss = -series
    var_95 = np.quantile(loss, 0.95)
    tail_count = 0.05 * len(loss)
    sorted_losses = np.sort(loss)[::-1]
    full = int(np.floor(tail_count))
    fractional = tail_count - full
    cvar_95 = (sorted_losses[:full].sum() + fractional * sorted_losses[full]) / tail_count
    return {
        "Доходность, годовых": float(series.mean() * periods),
        "Волатильность, годовых": float(series.std(ddof=1) * np.sqrt(periods)),
        "VaR 95%, за период": float(var_95),
        "CVaR 95%, за период": float(cvar_95),
        "Макс. просадка": float(-drawdown.min()),
    }


def optimize(returns: pd.DataFrame, lower: np.ndarray, upper: np.ndarray,
             annual_rf: float, periods: int) -> dict[str, np.ndarray | str]:
    """Follow Riskfolio-Lib tutorial 1: historical Classic/MV portfolio."""
    import riskfolio as rp

    n = returns.shape[1]
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")
    # Riskfolio-Lib uses A @ w <= B for linear inequalities.
    port.ainequality = np.vstack((-np.eye(n), np.eye(n)))
    port.binequality = np.r_[-lower, upper].reshape(-1, 1)
    results: dict[str, np.ndarray | str] = {}
    for label, objective in (("Минимальный риск", "MinRisk"),
                             ("Макс. доходность / риск", "Sharpe")):
        try:
            w = port.optimization(model="Classic", rm="MV", obj=objective,
                                  rf=annual_rf / periods, hist=True)
            if w is None:
                raise ValueError("решатель не нашёл решение")
            weights = w.reindex(returns.columns).iloc[:, 0].to_numpy(dtype=float)
            if (not np.isfinite(weights).all() or
                    not np.isclose(weights.sum(), 1, atol=1e-3) or
                    np.any(weights < lower - 1e-3) or np.any(weights > upper + 1e-3)):
                raise ValueError("решение нарушает заданные ограничения")
            results[label] = weights
        except Exception as exc:
            results[label] = str(exc)
    return results
