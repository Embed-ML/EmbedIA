from enum import IntEnum

class PaddingType(IntEnum):
    """
    Enumeration for padding types used across EmbedIA layers.
    
    Values are compatible with existing C code and layer implementations.
    """
    VALID = 0   # No padding
    SAME = 1    # Zero padding to maintain output size
    CAUSAL = 2  # Causal padding (only for 1D convolutions)