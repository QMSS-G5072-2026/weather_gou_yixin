import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from weather_yg3087 import (
    celsius_to_fahrenheit,
    summarize_temperatures,
    clean_readings,
)


@pytest.mark.parametrize(
    "celsius, expected",
    [
        (0, 32.0),
        (100, 212.0),
        (-40, -40.0),
        (37, 98.6),
        (-273.15, -459.67),
    ],
)
def test_celsius_to_fahrenheit(celsius, expected):
    """Test Celsius-to-Fahrenheit conversion for key temperatures."""
    result = celsius_to_fahrenheit(celsius)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad_input, expected_error",
    [
        ("37", TypeError),
        (None, TypeError),
        (True, TypeError),
        (-273.16, ValueError),
    ],
)
def test_celsius_to_fahrenheit_invalid(bad_input, expected_error):
    """Test invalid Celsius inputs raise the expected exception."""
    with pytest.raises(expected_error):
        celsius_to_fahrenheit(bad_input)


def test_summarize_temperatures():
    """Test summary statistics while ignoring one NaN value."""

    # Arrange
    temps = [10.0, float("nan"), 20.0]

    # Act
    result = summarize_temperatures(temps)

    # Assert
    assert result["mean"] == 15.0
    assert result["min"] == 10.0
    assert result["max"] == 20.0
    assert result["n_missing"] == 1


@pytest.fixture
def raw_readings():
    """Provide raw readings with missing and implausible temperatures."""
    return pd.DataFrame({
        "station": ["JFK", "CPK", "JFK", "CPK", "LGA"],
        "date": [
            "2026-07-02",
            "2026-07-01",
            "2026-07-01",
            "2026-07-02",
            "2026-07-01",
        ],
        "temp_c": [
            30.0,
            25.0,
            float("nan"),
            999.0,
            28.0,
        ],
    })


def test_raw_readings_fixture(raw_readings):
    """Check fixture columns, row count, and missing values."""
    assert list(raw_readings.columns) == [
        "station",
        "date",
        "temp_c",
    ]
    assert len(raw_readings) == 5
    assert raw_readings["temp_c"].isna().sum() == 1


def test_clean_readings(raw_readings):
    """Check that readings are cleaned, converted, sorted, and reindexed."""
    expected = pd.DataFrame({
        "station": ["CPK", "JFK", "LGA"],
        "date": [
            "2026-07-01",
            "2026-07-02",
            "2026-07-01",
        ],
        "temp_c": [25.0, 30.0, 28.0],
        "temp_f": [77.0, 86.0, 82.4],
    })

    result = clean_readings(raw_readings)

    assert_frame_equal(result, expected)