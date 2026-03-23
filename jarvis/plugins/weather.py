"""Weather lookup plugin."""

from __future__ import annotations

import json
import logging
from urllib.parse import quote
from urllib.request import urlopen

from jarvis.config import JarvisConfig
from jarvis.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class WeatherPlugin(BasePlugin):
    """Weather lookup actions."""

    def __init__(self, config: JarvisConfig):
        self._config = config

    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "get_weather",
                "description": (
                    "Get current weather or today's forecast for a location. "
                    "Use this instead of shell_command for weather requests."
                ),
                "parameters": {
                    "location": {
                        "type": "string",
                        "description": "Location to look up, e.g. 'Notre Dame, Indiana'",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Either 'current' or 'today'",
                        "default": "current",
                    },
                },
                "destructive": False,
                "handler": self.get_weather,
            }
        ]

    def get_weather(self, location: str, timeframe: str = "current") -> str:
        """Fetch weather from wttr.in for a location."""
        location = (location or "").strip()
        timeframe = (timeframe or "current").strip().lower()
        if not location:
            return "I need a location to check the weather."

        url = f"https://wttr.in/{quote(location)}?format=j1"
        logger.info("Fetching weather for %s (%s)", location, timeframe)

        try:
            with urlopen(url, timeout=10) as response:
                payload = json.load(response)
        except Exception as e:
            logger.error("Weather lookup failed for %s: %s", location, e)
            return f"I couldn't fetch the weather for {location}."

        current = payload.get("current_condition", [{}])[0]
        today = payload.get("weather", [{}])[0]
        nearest = payload.get("nearest_area", [{}])[0]

        area = _first_value(nearest.get("areaName"), location)
        region = _first_value(nearest.get("region"))
        country = _first_value(nearest.get("country"))
        resolved_location = ", ".join(part for part in [area, region or country] if part)
        display_location = location or resolved_location

        condition = _first_value(current.get("weatherDesc"), "unknown conditions")
        temp_f = current.get("temp_F")
        feels_f = current.get("FeelsLikeF")
        wind = current.get("windspeedMiles")
        humidity = current.get("humidity")

        if timeframe == "today":
            high_f = today.get("maxtempF")
            low_f = today.get("mintempF")
            return (
                f"{display_location}: {condition}, currently {temp_f} degrees. "
                f"Today looks like a high of {high_f} and a low of {low_f}."
            )

        bits = [f"{display_location}: {condition}"]
        if temp_f:
            bits.append(f"{temp_f} degrees")
        if feels_f:
            bits.append(f"feels like {feels_f}")
        if wind:
            bits.append(f"wind {wind} miles per hour")
        if humidity:
            bits.append(f"humidity {humidity} percent")
        return ", ".join(bits) + "."


def _first_value(entries, default: str = "") -> str:
    """Extract wttr.in [{value: ...}] fields."""
    if isinstance(entries, list) and entries:
        first = entries[0]
        if isinstance(first, dict):
            value = first.get("value")
            if isinstance(value, str):
                return value.strip()
    return default
