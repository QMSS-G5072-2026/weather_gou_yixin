import math

import pandas as pd


ABSOLUTE_ZERO_C = -273.15
MIN_PLAUSIBLE_C = -90.0
MAX_PLAUSIBLE_C = 60.0


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert a temperature from degrees Celsius to degrees Fahrenheit.

    Parameters
    ----------
    celsius : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Temperature in degrees Fahrenheit.

    Raises
    ------
    TypeError
        If `celsius` is not an int or a float.
    ValueError
        If `celsius` is below absolute zero (-273.15 C).

    Examples
    --------
    >>> import weather_yg3087
    >>> weather_yg3087.celsius_to_fahrenheit(0)
    32.0
    >>> weather_yg3087.celsius_to_fahrenheit(37)
    98.6
    """
    if isinstance(celsius, bool) or not isinstance(celsius, (int, float)):
        raise TypeError("celsius must be a number")

    if celsius < ABSOLUTE_ZERO_C:
        raise ValueError("temperature cannot be below absolute zero")

    return celsius * 9 / 5 + 32


def summarize_temperatures(temps: list[float]) -> dict[str, float]:
    """
    Summarize temperature readings while ignoring missing values.

    Parameters
    ----------
    temps : list of float
        Temperature readings. Missing readings are stored as float("nan").

    Returns
    -------
    dict
        A dictionary with the keys "mean", "min", "max", and "n_missing".

    Raises
    ------
    ValueError
        If there are no valid temperature readings.

    Examples
    --------
    >>> import weather_yg3087
    >>> weather_yg3087.summarize_temperatures(
    ...     [10.0, float("nan"), 20.0]
    ... )
    {'mean': 15.0, 'min': 10.0, 'max': 20.0, 'n_missing': 1}
    """
    valid = [t for t in temps if not math.isnan(t)]

    if not valid:
        raise ValueError("no valid temperature readings")

    return {
        "mean": sum(valid) / len(valid),
        "min": min(valid),
        "max": max(valid),
        "n_missing": len(temps) - len(valid),
    }


def clean_readings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw weather station readings.

    The function removes missing and implausible temperatures,
    converts Celsius values to Fahrenheit, and sorts the result.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns `station`, `date`, and `temp_c`.

    Returns
    -------
    pandas.DataFrame
        A new cleaned DataFrame containing the original columns plus
        `temp_f`, sorted by `station` and `date` with a reset index.

    Notes
    -----
    Temperature readings below -90 C or above 60 C are treated as
    physically implausible and removed. The boundary values -90 C
    and 60 C are retained.
    """
    cleaned = df.dropna(subset=["temp_c"])

    in_range = cleaned["temp_c"].between(
        MIN_PLAUSIBLE_C,
        MAX_PLAUSIBLE_C,
    )

    cleaned = cleaned[in_range].copy()

    cleaned["temp_f"] = cleaned["temp_c"] * 9 / 5 + 32

    return cleaned.sort_values(
        ["station", "date"]
    ).reset_index(drop=True)
