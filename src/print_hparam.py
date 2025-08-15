import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config.yaml")
def print_hparam(cfg:DictConfig):
    print(cfg)

if __name__ == "__main__":
    print_hparam()