from enum import Enum

class OpenNerfFieldHeadNames(Enum):
    """Possible field outputs"""
    HASHGRID = "hashgrid"
    DINO = "dino"
    OPENSEG = "openseg"