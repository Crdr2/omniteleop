"""Load the small amount of runtime configuration shared by leader and follower."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from omniteleop import LIB_PATH


class RobotConfig(dict[str, Any]):
    """YAML-backed topic configuration selected by ``ROBOT_CONFIG``."""

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            name = os.environ.get("ROBOT_CONFIG", "vega_1_gripper")
            config_path = LIB_PATH / "configs" / f"{name}.yaml"
        config_path = Path(config_path)
        if not config_path.is_file():
            available = sorted(path.stem for path in config_path.parent.glob("*.yaml"))
            raise FileNotFoundError(
                f"Robot config not found: {config_path}. Set ROBOT_CONFIG to one of {available}."
            )
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{config_path}: expected a top-level mapping")
        super().__init__(data)
        self.config_path = config_path
        logger.info(f"Loaded robot config from {config_path}")

    def get_topic(self, name: str, default: str | None = None) -> str | None:
        """Return a configured Zenoh topic."""
        return self.get("topics", {}).get(name, default)


_config: RobotConfig | None = None


def get_config(config_path: Path | None = None) -> RobotConfig:
    """Return the process-wide configuration, reloading for an explicit path."""
    global _config
    if config_path is not None or _config is None:
        _config = RobotConfig(config_path)
    return _config
