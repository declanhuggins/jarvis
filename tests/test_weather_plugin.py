"""Tests for the weather plugin."""

from jarvis.config import JarvisConfig
from jarvis.plugins.weather import WeatherPlugin


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_get_weather_current(monkeypatch):
    payload = b"""
    {
      "nearest_area": [{"areaName": [{"value": "Notre Dame"}], "region": [{"value": "Indiana"}]}],
      "current_condition": [{
        "temp_F": "42",
        "FeelsLikeF": "37",
        "weatherDesc": [{"value": "Cloudy"}],
        "windspeedMiles": "12",
        "humidity": "75"
      }],
      "weather": [{"maxtempF": "45", "mintempF": "31"}]
    }
    """
    monkeypatch.setattr("jarvis.plugins.weather.urlopen", lambda *args, **kwargs: _FakeResponse(payload))

    result = WeatherPlugin(JarvisConfig()).get_weather("Notre Dame, Indiana")

    assert "Notre Dame, Indiana: Cloudy" in result
    assert "42 degrees" in result


def test_get_weather_today(monkeypatch):
    payload = b"""
    {
      "nearest_area": [{"areaName": [{"value": "South Bend"}], "region": [{"value": "Indiana"}]}],
      "current_condition": [{
        "temp_F": "39",
        "FeelsLikeF": "33",
        "weatherDesc": [{"value": "Overcast"}]
      }],
      "weather": [{"maxtempF": "44", "mintempF": "29"}]
    }
    """
    monkeypatch.setattr("jarvis.plugins.weather.urlopen", lambda *args, **kwargs: _FakeResponse(payload))

    result = WeatherPlugin(JarvisConfig()).get_weather("South Bend, Indiana", timeframe="today")

    assert "high of 44" in result
    assert "low of 29" in result
