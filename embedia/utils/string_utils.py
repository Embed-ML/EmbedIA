import re
from collections import defaultdict


def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class NameGenerator:
    def __init__(self):
        self.clear()
        self._fmt = "%s_%d"

    def clear(self):
        self.names = defaultdict(int)

    def get(self, a_name):
        a_name = a_name.lower()
        self.names[a_name] += 1
        return self._fmt % (a_name, self.names[a_name])
