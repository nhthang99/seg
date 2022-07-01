import ignite
from ignite.engine import Events

from flame.module import Module


class TerminateOnNan(ignite.handlers.TerminateOnNan, Module):
    def init(self):
        self.frame['engine'].engine.add_event_handler(Events.ITERATION_COMPLETED, self)
