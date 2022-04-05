"""
Configuration constants used accross many files.
"""

from enum import Enum 

class NumLinks(Enum):
    TWO = 2
    FOUR = 4

base_path = "/Users/dani/Documents/lis/code/PaperManipulation/"

# TODO: should this be somewhere else? seems like it should be enforced elsewhere.
hinge_rotation_axis = 1
