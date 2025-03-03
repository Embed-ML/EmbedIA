
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
            return self._target.name
        if hasattr(self._target, '__name__'):
            return self._target.__name__
        if hasattr(self._target, '__class__'):
            return self._target.__class__.__name__
        return self.__class__.__name__.removesuffix('Wrapper')

    @property
    def input_shape(self):
        return None

    @property
    def output_shape(self):
        return None

