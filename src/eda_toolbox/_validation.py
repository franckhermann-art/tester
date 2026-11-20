"""Validations partagées des entrées tabulaires."""

import pandas as pd


def require_dataframe(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data doit être un pandas.DataFrame.")
