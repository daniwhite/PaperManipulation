"""
Constants used accross multiple project files.
Not expected to be changed frequently.
"""

from dataclasses import dataclass
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

PEDESTAL_X_DIM__2_LINKS = PLYWOOD_LENGTH/2
PEDESTAL_X_DIM__4_LINKS = PLYWOOD_LENGTH/4

Nm__PER_InLb = 0.112984833 # McMaster uses imperial units to a fault
stiffness__InLb_per_Rad__2_links = 6*0.55/(np.pi)
stiffness__InLb_per_Rad__4_links = 40/(np.pi)
stiffness_Nm_per_Rad__2_links = Nm__PER_InLb*stiffness__InLb_per_Rad__2_links
stiffness_Nm_per_Rad__4_links = Nm__PER_InLb*stiffness__InLb_per_Rad__4_links

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
    w_L = PEDESTAL_X_DIM__2_LINKS,
    h_L = THIN_PLYWOOD_THICKNESS,
    m_L = PLYWOOD_LENGTH*PEDESTAL_X_DIM__2_LINKS*\
        THIN_PLYWOOD_THICKNESS*PLYWOOD_DENSITY,
    m_M = 1e-3,
    b_J = 1e-1,
    k_J = stiffness_Nm_per_Rad__2_links,
    mu = 0.1,
    r = 0.05,
)
