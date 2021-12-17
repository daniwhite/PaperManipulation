from dataclasses import dataclass

@dataclass
class SystemConstants:
    """
    System constants
    """
    v_stiction: float
    I_L: float
    w_L: float
    h_L: float
    m_L: float
    m_M: float
    b_J: float
    k_J: float
    g: float
    mu_paper: float
    mu: float
    r: float
