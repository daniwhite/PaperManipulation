"""
Configuration constants used accross many files.
"""

from enum import Enum

# Time constants
DT = 0  # Means continuous time
TSPAN = 5 #0.001

# Number of links
class NumLinks(Enum):
    TWO = 2
    FOUR = 4
num_links = NumLinks.TWO
