"""Constants used accross multiple project files."""

# Standard imports
import numpy as np

# Imports of other project files
from pedestal import PEDESTAL_WIDTH


EPSILON = 0.001
IN_TO_M = 0.0254
FRICTION = 0.5

# Fingers normally are around 2cm wide
FINGER_RADIUS = 0.01
FINGER_VOLUME = (4/3)*FINGER_RADIUS**3*np.pi
FINGER_MASS = FINGER_VOLUME*1e3  # Assume finger is made of water

# Initial manipulator position
INIT_Y = PEDESTAL_WIDTH/2 + FINGER_RADIUS + EPSILON
INIT_Z = 0

# Time constants
DT = 1e-4  # 1e-5
TSPAN = 1
