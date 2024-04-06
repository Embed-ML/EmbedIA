
class UnsupportedLayerError(Exception):
    types_dict = {}

    def __init__(self, obj):
        msg = f"EmbedIA layer/element not implemented for {str(type(obj))}"
        super().__init__(msg)
        self.object = obj


class UnsupportedFeatureError(Exception):
    def __init__(self, obj, feature):
        super().__init__(
            f"EmbedIA feature ({feature}) not implemented for {str(type(obj))}"
            )
        self.object = obj
