"""
Constants used accross multiple project files.
Not expected to be changed frequently.
"""

from dataclasses import dataclass
import config
import numpy as np

g = 9.81

EPSILON = 0.001
IN_TO_M = 0.0254
v_stiction=1e-3

# Dimensions of physical pieces
THIN_PLYWOOD_THICKNESS = IN_TO_M*3/8
THICK_PLYWOOD_THICKNESS = IN_TO_M*3/4
PLYWOOD_LENGTH = IN_TO_M*12
PLYWOOD_DENSITY = 500 # kg/m^3
if config.num_links == config.NumLinks.TWO:
    PEDESTAL_X_DIM = PLYWOOD_LENGTH/2
elif config.num_links == config.NumLinks.FOUR:
    PEDESTAL_X_DIM = PLYWOOD_LENGTH/4

Nm__PER_InLb = 0.112984833 # McMaster uses imperial units to a fault
if config.num_links == config.NumLinks.TWO:
    stiffness__InLb_per_Rad = 6*0.55/(np.pi)
elif config.num_links == config.NumLinks.FOUR:
    stiffness__InLb_per_Rad = 40/(np.pi)
stiffness_Nm_per_Rad = Nm__PER_InLb*stiffness__InLb_per_Rad

@dataclass
class SystemConstants:
    """
    Constants for this particular plant + manipulator
    """
    w_L: float
    h_L: float
    m_L: float
    m_M: float
    b_J: float
    k_J: float
    mu: float
    r: float

nominal_sys_consts = SystemConstants(
    w_L = PEDESTAL_X_DIM,
    h_L = THIN_PLYWOOD_THICKNESS,
    m_L = PLYWOOD_LENGTH*PEDESTAL_X_DIM*THIN_PLYWOOD_THICKNESS*PLYWOOD_DENSITY,
    m_M = 1e-3,
    b_J = 1e-1,
    k_J = stiffness_Nm_per_Rad,
    mu = 0.2,
    r = 0.05,
)
