"""
Configuration constants used accross many files.
"""

from enum import Enum

# Time constants
DT = 0.0001
TSPAN = 10

base_path = "/Users/dani/Documents/lis/code/PaperManipulation/"

# Number of links
class NumLinks(Enum):
    TWO = 2
    FOUR = 4
num_links = NumLinks.TWO

# TODO: should this be somewhere else? seems like it should be enforced elsewhere.
hinge_rotation_axis = 1
