import yaml

def read_config(config_path: str = "config.yaml") -> dict:
    """Read the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration data.
    """
    config_path = config_path or "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config