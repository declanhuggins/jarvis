"""Music playback plugin."""

from __future__ import annotations

import logging
import subprocess
from urllib.parse import quote

from jarvis.config import JarvisConfig
from jarvis.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class MusicPlugin(BasePlugin):
    """Spotify playback actions."""

    def __init__(self, config: JarvisConfig):
        self._config = config

    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "play_spotify",
                "description": (
                    "Play music in Spotify. If the user names a playlist, album, artist, or mix "
                    "such as Daylist, pass that in the query field."
                ),
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to play in Spotify, e.g. 'my daylist', 'lofi beats', "
                            "'spotify:playlist:...'. Use an empty string to resume playback."
                        ),
                        "default": "",
                    }
                },
                "destructive": False,
                "handler": self.play_spotify,
            },
            {
                "name": "pause_spotify",
                "description": "Pause Spotify playback",
                "parameters": {},
                "destructive": False,
                "handler": self.pause_spotify,
            },
            {
                "name": "next_spotify_track",
                "description": "Skip to the next Spotify track",
                "parameters": {},
                "destructive": False,
                "handler": self.next_track,
            },
            {
                "name": "previous_spotify_track",
                "description": "Go to the previous Spotify track",
                "parameters": {},
                "destructive": False,
                "handler": self.previous_track,
            },
        ]

    def play_spotify(self, query: str = "") -> str:
        """Open or resume Spotify playback."""
        query = (query or "").strip()
        logger.info("Playing Spotify query: %s", query or "(resume)")

        if query:
            target = _spotify_target_for_query(query)
            subprocess.run(["open", target], check=True, capture_output=True)

        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "Spotify" to activate',
                "-e",
                "delay 1",
                "-e",
                'tell application "Spotify" to play',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if query:
            return f"Playing {query} on Spotify"
        return "Resumed Spotify"

    def pause_spotify(self) -> str:
        """Pause Spotify playback."""
        subprocess.run(
            ["osascript", "-e", 'tell application "Spotify" to pause'],
            check=True,
            capture_output=True,
            text=True,
        )
        return "Paused Spotify"

    def next_track(self) -> str:
        """Skip to the next Spotify track."""
        subprocess.run(
            ["osascript", "-e", 'tell application "Spotify" to next track'],
            check=True,
            capture_output=True,
            text=True,
        )
        return "Skipped to the next Spotify track"

    def previous_track(self) -> str:
        """Go back to the previous Spotify track."""
        subprocess.run(
            ["osascript", "-e", 'tell application "Spotify" to previous track'],
            check=True,
            capture_output=True,
            text=True,
        )
        return "Went back to the previous Spotify track"


def _spotify_target_for_query(query: str) -> str:
    """Map a free-text Spotify request to a URI or search target."""
    if query.startswith("spotify:"):
        return query
    if query.startswith("https://open.spotify.com/"):
        return query
    return f"spotify:search:{quote(query)}"
