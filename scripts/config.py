"""
config.py

Load and validate experiment configurations.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    name: str
    data_dir: str
    batch_size: int
    learning_rate: float
    early_stop_count: int
    epochs: int
    model: str
    scheduler_type: str
    lambda_dice: float
    lambda_ce: float

def load_config(experiment: str) -> ExperimentConfig:
    config_path = Path("results") / experiment / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    lp = raw["loss_parameters"]
    return ExperimentConfig(
        name=raw["experiment"],
        data_dir=raw["data_dir"],
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        early_stop_count=raw["early_stop_count"],
        epochs=raw["epochs"],
        model=raw["model"],
        scheduler_type=raw.get("scheduler_type", "plateau"),
        lambda_dice=lp["lambda_dice"],
        lambda_ce=lp["lambda_ce"],
    )