# -*- coding: utf-8 -*-
#
# Gary Hammock, 2022
# SPDX-License-Identifier: MIT
#
# This script shows the massive speed up between using Pandas/NumPy `apply()` and
# using vectorization.

from functools import wraps
from typing import Callable

from line_profiler import LineProfiler  # type: ignore  # line_profiler doesn't provide type hints.
import numpy as np
import pandas as pd  # type: ignore  # pandas doesn't provide type hints (yet).


def profile(func: Callable) -> Callable:
    """
    Create a decorator for wrapping a provided function in a LineProfiler context.

    Parameters
    ----------
    func : callable
        The function that is to be wrapped inside the LineProfiler context.

    Returns
    -------
    wrapper : callable
        The context containing the wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()
    return wrapper


def round_to_nearest_point_5(value: float) -> float:
    """
    Rounds each top value to the nearest increment of 0.5

    Parameters
    ----------
    value : int
        each individual top value given

    Returns
    -------
    float
        A new top value expressed to the nearest 0.5 increment

    Notes
    -----
    This is surprisingly performant on scalar values.  Division is typically an
    expensive operation, but the multiply, round, and divide operations are at least
    twice as fast as other approaches (e.g. using if-statements with ``divmod()``).
    """

    return round(2 * value) / 2


def generate_dataframe() -> pd.DataFrame:
    """
    Generates a sample DataFrame with one-million elements in a single Series.

    The values of Series are randomly selected from a uniform distribution and are
    sorted in ascending order. i.e. for i in {i ∈ ℤ | 0 ≤ i < 1,000,000},
    x[i] = z where {z ∈ ℝ | 100 ≤ z < 100,000}.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (10⁶, 1) containing only the column "MD_TOP".

    """

    return pd.DataFrame({
        "MD_TOP": np.sort(np.random.uniform(low=100, high=100_000, size=1_000_000))
    })


# This is the DataFrame whose Series values are used in the ``apply()`` method.
apply_df: pd.DataFrame = generate_dataframe()

# This is a deep-copy clone of the generated DataFrame whose values will be
# modified using Pandas/NumPy native vectorization.
vectorize_df: pd.DataFrame = apply_df.copy(deep=True)


@profile
def driver() -> None:
    """
    Apply the different methods and ensure they produce the same result.

    This is an instrumented function to measure the performance of the DataFrame
    modifications through ``pd.DataFrame.apply()`` and a vectorized form of the
    ``round_to_nearest_point_5()`` function.

    See Also
    --------
    round_to_nearest_point_5

    Returns
    -------
    None

    Notes
    -----
    Repeated tests show the vectorization method

    """
    apply_df["MD_TOP"] = apply_df["MD_TOP"].apply(round_to_nearest_point_5)
    vectorize_df["MD_TOP"] = (vectorize_df["MD_TOP"] * 2).round() / 2

    assert apply_df.equals(vectorize_df), "The DataFrames are NOT equal."


if __name__ == "__main__":
    driver()
