import pydrake
from pydrake.all import BodyIndex

import numpy as np

from dataclasses import dataclass

@dataclass
class IndependentParams:
    w_L: float
    h_L: float
    m_L: float
    m_M: float
    b_J: float
    k_J: float
    mu: float
    r: float

@dataclass
class DependentParams:
    I_L: float

import config

import plant.pedestal as pedestal
import plant.manipulator as manipulator
from plant.paper import Paper

import constants

class Simulation:
    def __init__(self, params=None, nominal_joint_angles=0):
        if params is None:
            params = constants.nominal_sys_consts

        # Multibody plant setup
        self.builder = pydrake.systems.framework.DiagramBuilder()

        # Add all elements
        self.plant, self.scene_graph = \
            pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
                self.builder, time_step=config.DT)
        self.plant.set_stiction_tolerance(constants.v_stiction)
        self.plant.set_penetration_allowance(0.001)

        # Add other systems
        pedestal.AddPedestal(self.plant)

        self.paper = Paper(self.plant, self.scene_graph,
            default_joint_angle=nominal_joint_angles,
            k_J=params.k_J, b_J=params.b_J,
            m_L=params.m_L, w_L=params.w_L, h_L=params.h_L, mu=params.mu)
        self.paper.weld_paper_edge(
            constants.PEDESTAL_Y_DIM, pedestal.PEDESTAL_Z_DIM)

        self.manipulator_instance = manipulator.data["add_plant_function"](
            plant=self.plant, m_M=params.m_M, r=params.r, mu=params.mu,
            scene_graph=self.scene_graph)
        self.plant.Finalize()

        # TODO: Does this need to be an attribute? Should this be earlier?
        self.params = params

        # Initialize sys_consts for other systems
        self.sys_consts = constants.SystemConstants
        self.sys_consts.I_L = self.plant.get_body(
            BodyIndex(self.paper.link_idxs[-1])).default_rotational_inertia(
                ).CalcPrincipalMomentsOfInertia()[0]
        self.sys_consts.v_stiction = constants.v_stiction
        self.sys_consts.w_L = self.params.w_L
        self.sys_consts.h_L = self.params.h_L
        self.sys_consts.m_L = self.params.m_L
        self.sys_consts.m_M = self.params.m_M
        self.sys_consts.b_J = self.params.b_J
        self.sys_consts.k_J = self.params.k_J
        self.sys_consts.mu = self.params.mu
        self.sys_consts.r = self.params.r
        self.sys_consts.g = self.plant.gravity_field().gravity_vector()[-1]*-1

        # TODO: Calculate ll_idx, contact_body_idx

