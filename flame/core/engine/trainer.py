import torch
import torch.nn as nn

from flame.core.engine.engine import Engine


class Trainer(Engine):
    '''
        Engine controls training process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'loss' in self.frame, 'The frame does not have loss.'
        self.model = self.frame['model'].to(self.device)
        if torch.cuda.device_count() > 1 and self.device != 'cpu':
            self.model = nn.DataParallel(self.model)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']
        print("Total parameters:", sum(p.numel() for p in self.model.parameters()))
        print("Number of learnable parameters:", sum(p.numel() for p in self.model.parameters()
                                                     if p.requires_grad))
        print("Number of non-learnable parameters:", sum(p.numel() for p in self.model.parameters()
                                                         if not p.requires_grad))

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
        params[0] = self.model(params[0])
        loss = self.loss(*params)
        loss.backward()
        self.optimizer.step()
        return loss.item()
