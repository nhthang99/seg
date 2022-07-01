from abc import ABC, abstractmethod


class Frame(dict):
    def __init__(self, config_path: str):
        super(Frame, self).__init__()
        self.config_path = config_path


class ModuleBase(ABC):
    """Abstract module
    Submodules should be inherited from this module
    """
    def attach(self, frame: Frame, module_name: str):
        """Attach a submodule into a module
        Args:
            frame (_type_): _description_
            module_name (_type_): _description_
        """
        self.frame = frame
        self.frame[module_name] = self
        self.module_name = module_name

    @abstractmethod
    def init(self):
        pass
