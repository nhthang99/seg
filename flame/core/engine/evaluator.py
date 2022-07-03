import torch
import torch.nn as nn

from .engine import Engine
from ignite.engine import Events


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)
        if torch.cuda.device_count() > 1 and self.device != 'cpu':
            self.model = nn.DataParallel(self.model)
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self.run)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            params[0] = self.model(params[0])
            return params
