from flame.core.engine.evaluator import Evaluator

from ignite.engine import Events


class MetricEvaluator(Evaluator):
    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.model = self.frame['model'].to(self.device)
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._run)

    def _run(self, engine):
        self.run()
