"""File management plugin: organize downloads, desktop, and search files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from jarvis.config import JarvisConfig
from jarvis.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

# File extension to category mapping
CATEGORIES: dict[str, set[str]] = {
    "Images": {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
        ".webp", ".heic", ".tiff", ".ico", ".raw",
    },
    "Documents": {
        ".pdf", ".doc", ".docx", ".txt", ".rtf", ".pages",
        ".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".key",
        ".numbers", ".odt", ".ods", ".odp",
    },
    "Archives": {
        ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
        ".dmg", ".iso", ".pkg",
    },
    "Videos": {
        ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv",
        ".webm", ".m4v",
    },
    "Audio": {
        ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg",
        ".wma", ".aiff",
    },
    "Code": {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
        ".json", ".yaml", ".yml", ".sh", ".bash", ".zsh",
        ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb",
        ".swift", ".kt", ".sql", ".md", ".xml", ".toml",
    },
    "Installers": {
        ".app", ".dmg", ".pkg", ".deb", ".rpm", ".msi", ".exe",
    },
}


def _categorize(ext: str) -> str:
    """Map a file extension to a category name."""
    ext = ext.lower()
    for category, extensions in CATEGORIES.items():
        if ext in extensions:
            return category
    return "Other"


class FilesPlugin(BasePlugin):
    """File organization and search actions."""

    def __init__(self, config: JarvisConfig):
        self._config = config

    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "cleanup_downloads",
                "description": (
                    "Organize ~/Downloads by moving files into subfolders "
                    "by type (Images, Documents, Archives, Videos, Audio, Code, etc.)"
                ),
                "parameters": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, report what would be moved without actually moving files",
                        "default": False,
                    }
                },
                "destructive": True,
                "handler": self.cleanup_downloads,
            },
            {
                "name": "cleanup_desktop",
                "description": (
                    "Organize ~/Desktop by moving files into subfolders "
                    "by type (Images, Documents, Archives, etc.)"
                ),
                "parameters": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, report what would be moved without actually moving files",
                        "default": False,
                    }
                },
                "destructive": True,
                "handler": self.cleanup_desktop,
            },
            {
                "name": "find_files",
                "description": "Search for files matching a name pattern in a directory",
                "parameters": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename glob pattern, e.g. '*.pdf', 'report*'",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: home directory)",
                        "default": "~",
                    },
                },
                "destructive": False,
                "handler": self.find_files,
            },
            {
                "name": "move_file",
                "description": "Move a file or folder from one location to another",
                "parameters": {
                    "source": {
                        "type": "string",
                        "description": "Source file or folder path",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path",
                    },
                },
                "destructive": True,
                "handler": self.move_file,
            },
            {
                "name": "trash_file",
                "description": "Move a file or folder to the macOS Trash (recoverable)",
                "parameters": {
                    "path": {
                        "type": "string",
                        "description": "File or folder path to trash",
                    },
                },
                "destructive": True,
                "handler": self.trash_file,
            },
        ]

    def cleanup_downloads(self, dry_run: bool = False) -> str:
        """Organize ~/Downloads into category subfolders."""
        return self._organize_directory(Path.home() / "Downloads", dry_run)

    def cleanup_desktop(self, dry_run: bool = False) -> str:
        """Organize ~/Desktop into category subfolders."""
        return self._organize_directory(Path.home() / "Desktop", dry_run)

    def _organize_directory(self, directory: Path, dry_run: bool) -> str:
        """Move files in a directory into category subfolders.

        Skips hidden files, directories, and files already in category folders.
        """
        if not directory.exists():
            return f"Directory {directory} does not exist"

        moved: dict[str, list[str]] = {}
        errors: list[str] = []

        # Get the set of category folder names to skip
        category_names = set(CATEGORIES.keys()) | {"Other"}

        for item in sorted(directory.iterdir()):
            # Skip hidden files, directories, and category folders themselves
            if item.name.startswith("."):
                continue
            if item.is_dir():
                continue

            category = _categorize(item.suffix)
            dest_dir = directory / category

            if not dry_run:
                try:
                    dest_dir.mkdir(exist_ok=True)
                    dest_file = dest_dir / item.name

                    # Handle name collisions
                    if dest_file.exists():
                        stem = item.stem
                        suffix = item.suffix
                        counter = 1
                        while dest_file.exists():
                            dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    shutil.move(str(item), str(dest_file))
                    logger.info("Moved %s -> %s", item.name, dest_file)
                except OSError as e:
                    errors.append(f"{item.name}: {e}")
                    logger.error("Failed to move %s: %s", item.name, e)
                    continue

            moved.setdefault(category, []).append(item.name)

        if not moved:
            return f"No files to organize in {directory.name}"

        prefix = "Would move" if dry_run else "Moved"
        lines = [
            f"{prefix} {len(files)} file{'s' if len(files) != 1 else ''} to {cat}"
            for cat, files in sorted(moved.items())
        ]
        summary = ". ".join(lines)

        if errors:
            summary += f". {len(errors)} file{'s' if len(errors) != 1 else ''} failed"

        return summary

    def find_files(self, pattern: str, directory: str = "~") -> str:
        """Search for files matching a glob pattern."""
        base = Path(directory).expanduser()
        if not base.exists():
            return f"Directory {directory} does not exist"

        logger.info("Searching for '%s' in %s", pattern, base)

        try:
            matches = list(base.rglob(pattern))
        except OSError as e:
            return f"Search error: {e}"

        # Cap results to avoid overwhelming output
        total = len(matches)
        matches = matches[:20]

        if not matches:
            return f"No files matching '{pattern}' found in {directory}"

        names = [str(m.relative_to(base)) for m in matches]
        result = f"Found {total} file{'s' if total != 1 else ''}"
        if total > 20:
            result += f" (showing first 20)"
        result += ": " + ", ".join(names)

        return result

    def move_file(self, source: str, destination: str) -> str:
        """Move a file or folder to a new location."""
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        if not src.exists():
            return f"Source not found: {source}"

        # If destination is a directory, move into it
        if dst.is_dir():
            dst = dst / src.name

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            logger.info("Moved %s -> %s", src, dst)
            return f"Moved {src.name} to {dst}"
        except OSError as e:
            return f"Failed to move: {e}"

    def trash_file(self, path: str) -> str:
        """Move a file to macOS Trash (recoverable via Finder)."""
        target = Path(path).expanduser()
        if not target.exists():
            return f"File not found: {path}"

        logger.info("Trashing: %s", target)

        # Use macOS Finder scripting to move to Trash (supports undo)
        import subprocess

        script = f'''
        tell application "Finder"
            delete POSIX file "{target.resolve()}"
        end tell
        '''
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
            )
            return f"Moved {target.name} to Trash"
        except subprocess.CalledProcessError as e:
            return f"Failed to trash {target.name}: {e.stderr.decode().strip()}"
