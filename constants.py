import numpy as np

EPSILON = 0.001
IN_TO_M = 0.0254
FRICTION = 0.5

# Fingers normally are around 2cm wide
FINGER_RADIUS = 0.01
FINGER_VOLUME = (4/3)*FINGER_RADIUS**3*np.pi
FINGER_MASS = FINGER_VOLUME*1e3  # Assume finger is made of water
