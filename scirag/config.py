import yaml
import os
from pathlib import Path

def get_config():
    path_file = Path(__file__)
    path_config = os.path.join(path_file.parent, "config.yaml")
    with open(path_config) as file:
        config= yaml.safe_load(file)

    for path in config:
        if path.startswith("path"):
            config[path] =  os.path.join(path_file.parent.parent, config[path])
    return config


