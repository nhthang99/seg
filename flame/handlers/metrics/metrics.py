import numpy as np
from ignite.engine import Events

from flame.module import Module


class Metrics(Module):
    def __init__(self, metrics, attach_to=None):
        super(Metrics, self).__init__()
        self.metrics = metrics
        self.train_loss = []
        self.metric_values = {}
        self.attach_to = attach_to if attach_to else {}

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        assert all(map(lambda x: x in self.frame, self.attach_to.keys())), \
               f'The frame does not have all {self.attach_to.keys()}.'
        self.frame['engine'].engine.add_event_handler(Events.ITERATION_COMPLETED,
                                                      self._save_train_step_result)
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED,
                                                      self._save_train_result, eval_name="train")
        for evaluator, eval_name in self.attach_to.items():
            evaluator = self.frame[evaluator]

            for metric_name, metric in self.metrics.items():
                metric.attach(evaluator.engine, metric_name)

            evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_eval_result, eval_name)

    def _save_eval_result(self, engine, eval_name):
        self.metric_values[eval_name] = engine.state.metrics

    def _save_train_step_result(self, engine):
        self.train_loss.append(engine.state.output)

    def _save_train_result(self, engine, eval_name):
        self.metric_values[eval_name] = {"loss": np.mean(self.train_loss).item()}
        self.train_loss = []
