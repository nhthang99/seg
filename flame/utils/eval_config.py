from typing import Any

import yaml
from importlib import import_module


def load_yaml(yaml_file):
    with open(yaml_file) as f:
        settings = yaml.safe_load(f)
    return settings


def eval_config(config: Any):
    """Eval config
    Args:
        config (_type_): _description_
    """
    def _eval_config(config):
        if isinstance(config, dict):
            global_extra_libs.update({name: import_module(lib)
                                    for (name, lib) in config.pop('extralibs', {}).items()})
            for key, value in config.items():
                if key not in ['module', 'class']:
                    config[key] = _eval_config(value)

            if 'module' in config and 'class' in config:
                module = config['module']
                class_ = config['class']
                is_instance = config.get('is_instance', True)
                config_kwargs = config.get(class_, {})
                if is_instance:
                    return getattr(import_module(module), class_)(**config_kwargs)
                else:
                    return getattr(import_module(module), class_)

            return config
        elif isinstance(config, list):
            return [_eval_config(ele) for ele in config]
        elif isinstance(config, str):
            return eval(config, global_extra_libs)
        else:
            return config

    global_extra_libs = {"config": config}
    configs = {}
    configs.update(_eval_config(config))
    if isinstance(config, dict) and "__base__" in config.keys():
        base_config =config.pop("__base__")
        if isinstance(base_config, str):
            base_config = [base_config]
        for base_cfg in base_config:
            with open(base_cfg) as f:
                cfg = yaml.safe_load(f)
                configs.update(_eval_config(cfg))

    return configs
