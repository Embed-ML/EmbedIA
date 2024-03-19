
from abc import abstractmethod

class LayerWrapper:

    def __init__(self, target: object):
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def name(self):
        if hasattr(self._target, 'name'):
            return self._.target.name
        return self.target.__name__

    @property
    def input_shape(self):
        return None

    @property
    def output_shape(self):
        return None

