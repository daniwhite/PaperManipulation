import numpy as np
import constants
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, PidController


class Paper:
    name = "paper"
    width = 11*constants.IN_TO_M
    depth = 8.5*constants.IN_TO_M
    height = 0.007

    # Source:
    # https://www.jampaper.com/paper-weight-chart.asp
    # height = 0.004*constants.IN_TO_M
    density = 80/1000

    # Source:
    # https://smartech.gatech.edu/bitstream/handle/1853/5562/jones_ar.pdf
    youngs_modulus = 10*1e9  # Convert to n/m^2

    stiffness = 0.00001
    damping = 0.00001

    def __init__(self, plant, num_links, mu=5.0, default_joint_angle=-np.pi/60):
        # Initialize parameters
        self.plant = plant
        self.num_links = num_links
        self.mu = mu
        self.default_joint_angle = default_joint_angle

        self.link_width = self.width/self.num_links
        self.link_mass = self.density*self.width*self.width/self.num_links

        L = self.width/num_links
        I = self.depth*self.height**3/12
        # Stiffness = 3*(youngs modulus)*I/(Length)
        physical_stiffness_N_p_m = 3*self.youngs_modulus*I/L**3
        physical_N_p_rad = physical_stiffness_N_p_m*L**2

        self.paper_instances = []
        for link_num in range(self.num_links):
            # Initalize bodies and instances
            paper_instance = self.plant.AddModelInstance(
                self.name + str(link_num))
            paper_body = self.plant.AddRigidBody(
                self.name + "_body" + str(link_num),
                paper_instance,
                pydrake.multibody.tree.SpatialInertia(mass=self.link_mass, p_PScm_E=np.array([0., 0., 0.]),
                                                      G_SP_E=pydrake.multibody.tree.UnitInertia.SolidBox(self.link_width, self.width, self.height))
            )

            # Set up collision
            if self.plant.geometry_source_is_registered():
                self.plant.RegisterCollisionGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.link_width, self.width, self.height),
                    self.name + "_body" +
                    str(link_num), pydrake.multibody.plant.CoulombFriction(
                        self.mu, self.mu)
                )
#                 i=0
#                 for x in [-self.link_width/2.0, self.link_width/2.0]:
#                     for y in [-self.width/2.0, self.width/2.0]:
#                         for z in [-self.height/2.0, self.height/2.0]:
#                             self.plant.RegisterCollisionGeometry(
#                                 paper_body,
#                                 RigidTransform([x, y, z]),
#                                 pydrake.geometry.Sphere(radius=1e-7), f"contact_sphere{i}_{link_num}",
#                                 pydrake.multibody.plant.CoulombFriction(self.mu, self.mu))
#                             i += 1
                self.plant.RegisterVisualGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.link_width, self.width, self.height),
                    self.name + "_body" + str(link_num),
                    [0.9, 0.9, 0.9, 1.0])

            # Set up joint actuators
            if link_num > 0:
                paper1_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    self.plant.GetBodyByName("paper_body{}".format(
                        link_num-1), self.paper_instances[-1]),
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

                joint.set_default_angle(self.default_joint_angle)
                ja = self.plant.AddJointActuator("joint actuator", joint)

            self.paper_instances.append(paper_instance)

    def init_ctrlrs(self, builder):
        """Initialize controllers for paper joints"""
        self.paper_ctrlrs = []
        for paper_instance in self.paper_instances[1:]:
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

    def weld_paper_edge(self, pedestal_width, pedestal_height):
        # Fix paper to object
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(
                "paper_body0", self.paper_instances[0]).body_frame(),
            RigidTransform(RotationMatrix(
            ), [-(pedestal_width/2-self.width/4), 0, pedestal_height+self.height/2])
        )
