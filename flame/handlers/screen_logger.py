import time

from ignite.engine import Events

from flame.module import Module


class ScreenLogger(Module):
    def __init__(self, eval_names=None):
        super(ScreenLogger, self).__init__()
        self.eval_names = eval_names if eval_names else []

    def _started(self, engine):
        msg = f'{time.asctime()} - STARTED'
        print(msg)

    def _completed(self, engine):
        msg = f'{time.asctime()} - COMPLETED'
        print(msg)

    def _log_screen(self, engine):
        msg = f'Epoch #{engine.state.epoch:3d} - {time.asctime()} - '
        for eval_name in self.eval_names:
            if eval_name not in self.frame['metrics'].metric_values.keys():
                continue
            for metric_name, metric_value in self.frame['metrics'].metric_values[eval_name].items():
                msg += f'{eval_name}_{metric_name}: {metric_value:.5f} - '
        print(msg[:-2])

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.STARTED, self._started)
        self.frame['engine'].engine.add_event_handler(Events.COMPLETED, self._completed)
        if len(self.eval_names):
            assert 'metrics' in self.frame, 'The frame does not have metrics.'
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_screen)
