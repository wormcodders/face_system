import yaml
from gui import launch_app

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

if __name__ == "__main__":
    launch_app(CONFIG)
