import sys
from pathlib import Path

import yaml

PYTHON_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(PYTHON_PATH))

from flame.utils.eval_config import eval_config
from flame.module import Module, Frame


if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frame = Frame(str(config_path))

    config = eval_config(config)
    for module_name, module in config.items():
        if isinstance(module, Module):
            module.attach(frame, module_name)
        else:
            frame[module_name] = module

    for module in frame.values():
        if isinstance(module, Module):
            module.init()

    assert 'trainer' in frame, 'The frame does not have trainer.'
    state = frame['trainer'].run()
