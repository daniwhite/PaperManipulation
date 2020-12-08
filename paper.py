import numpy as np
import constants
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, PidController
from pydrake.multibody.tree import SpatialInertia, UnitInertia


class Paper:
    name = "paper"
    width = 11*constants.IN_TO_M
    depth = 8.5*constants.IN_TO_M
    height = 0.0005

    # Source:
    # https://www.jampaper.com/paper-weight-chart.asp
    true_height = 0.0097e-3
    density = 80/1000

    # Source:
    # https://smartech.gatech.edu/bitstream/handle/1853/5562/jones_ar.pdf
    # http://www.mate.tue.nl/mate/pdfs/10509.pdf
    youngs_modulus = 6*1e9  # Convert to n/m^2

    def __init__(self, plant, num_links, mu=5.0, default_joint_angle=-np.pi/60, damping=1e-5, stiffness=1e-3):
        # Initialize parameters
        self.plant = plant
        self.num_links = num_links
        self.mu = constants.FRICTION
        self.default_joint_angle = default_joint_angle
        self.damping = damping
        self.stiffness = stiffness

        self.link_width = self.width/self.num_links
        self.link_mass = self.density*self.width*self.width/self.num_links

        L = self.width/num_links
        I = self.depth*self.true_height**3/12
        # Stiffness = 3*(youngs modulus)*I/(Length)
        physical_stiffness_N_p_m = 3*self.youngs_modulus*I/L**3
        physical_N_p_rad = physical_stiffness_N_p_m*L**2

        # Use this stiffness only if simulating with a shorter DT
        # self.stiffness = physical_N_p_rad

        self.link_instances = []
        for link_num in range(self.num_links):
            # Initalize bodies and instances
            paper_instance = self.plant.AddModelInstance(
                self.name + str(link_num))
            paper_body = self.plant.AddRigidBody(
                self.name + "_body" + str(link_num),
                paper_instance,
                SpatialInertia(mass=self.link_mass,
                               # CoM at origin of body frame
                               p_PScm_E=np.array([0., 0., 0.]),
                               # Deault moment of inertia for a solid box
                               G_SP_E=UnitInertia.SolidBox(
                                   self.link_width, self.width, self.true_height))
            )

            if self.plant.geometry_source_is_registered():
                # Set up collision geometery
                self.plant.RegisterCollisionGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.link_width, self.width, self.height),
                    self.name + "_body" +
                    str(link_num), pydrake.multibody.plant.CoulombFriction(
                        self.mu, self.mu)
                )

                # Set up visual geometry
                self.plant.RegisterVisualGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.link_width, self.width, self.height),
                    self.name + "_body" + str(link_num),
                    [0.9, 0.9, 0.9, 1.0])  # RBGA color

            # Set up joint actuators
            if link_num > 0:
                paper1_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    self.plant.GetBodyByName("paper_body{}".format(
                        link_num-1), self.link_instances[-1]),
                    RigidTransform(RotationMatrix(), [self.link_width/2+constants.EPSILON/2, 0, 0.5*self.height]))
                self.plant.AddFrame(paper1_hinge_frame)
                paper2_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    self.plant.GetBodyByName(
                        "paper_body{}".format(link_num), paper_instance),
                    RigidTransform(RotationMatrix(), [(-self.link_width/2+constants.EPSILON/2), 0, 0.5*self.height]))
                self.plant.AddFrame(paper2_hinge_frame)

                joint = self.plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
                    "paper_hinge",
                    paper1_hinge_frame,
                    paper2_hinge_frame,
                    [0, 1, 0]))

                if type(default_joint_angle) is list:
                    print(self.default_joint_angle[link_num])
                    joint.set_default_angle(self.default_joint_angle[link_num])
                else:
                    joint.set_default_angle(self.default_joint_angle)
                self.plant.AddJointActuator("joint actuator", joint)

            self.link_instances.append(paper_instance)

    def init_ctrlrs(self, builder):
        """Initialize controllers for paper joints"""
        self.paper_ctrlrs = []
        for paper_instance in self.link_instances[1:]:
            # In an ideal world, this would be done with a custom force element.
            # But since this is in Python, we can't implement the C++ class.
            paper_ctrlr = PidController(kp=[[self.stiffness]], ki=[
                                        [0]], kd=[[self.damping]])
            builder.AddSystem(paper_ctrlr)
            self.paper_ctrlrs.append(paper_ctrlr)
            builder.Connect(paper_ctrlr.get_output_port(),
                            self.plant.get_actuation_input_port(paper_instance))
            builder.Connect(self.plant.get_state_output_port(
                paper_instance), paper_ctrlr.get_input_port_estimated_state())

    def connect_ctrlrs(self, diagram, diagram_context):
        """Fix controller input to an angle of zero."""
        for paper_ctrlr in self.paper_ctrlrs:
            paper_ctrlr_context = diagram.GetMutableSubsystemContext(
                paper_ctrlr, diagram_context)
            paper_ctrlr.get_input_port_desired_state().FixValue(
                paper_ctrlr_context, [0, 0])

    def get_free_edge_instance(self):
        # return self.plant.GetModelInstanceByName(self.name + str(self.num_links-1))
        return self.link_instances[-2]

    def weld_paper_edge(self, pedestal_width, pedestal_height):
        # Fix paper to object
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(
                "paper_body0", self.link_instances[0]).body_frame(),
            RigidTransform(RotationMatrix(
            ), [-(pedestal_width/2-self.width/4), 0, pedestal_height+self.height/2])
        )
