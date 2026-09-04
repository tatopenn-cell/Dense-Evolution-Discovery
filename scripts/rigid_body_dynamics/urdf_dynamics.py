import xml.etree.ElementTree as ET

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

G = 9.81


def _floats(s):
    return [float(x) for x in s.split()]


def _vec3(el, tag, default="0 0 0"):
    sub = el.find(tag)
    return np.array(_floats(sub if isinstance(sub, str) else (sub.get("xyz") if sub is not None else default)))


def parse_urdf(path):
    """
    Parse a real URDF file into links/joints. No robot-specific knowledge here --
    this reads whatever <link>/<joint> elements are present, for any kinematic
    tree (not just a single chain): branching robots (e.g. two arms, a gripper
    with independent fingers) are supported via the parent/child adjacency built
    below, not assumed to be a single serial chain.
    """
    root = ET.parse(path).getroot()

    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        inertial = link_el.find("inertial")
        if inertial is None:
            links[name] = dict(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3)))
            continue
        origin = inertial.find("origin")
        xyz = np.array(_floats(origin.get("xyz", "0 0 0"))) if origin is not None else np.zeros(3)
        mass = float(inertial.find("mass").get("value"))
        i_el = inertial.find("inertia")
        ixx, ixy, ixz = float(i_el.get("ixx")), float(i_el.get("ixy")), float(i_el.get("ixz"))
        iyy, iyz, izz = float(i_el.get("iyy")), float(i_el.get("iyz")), float(i_el.get("izz"))
        inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        links[name] = dict(mass=mass, com=xyz, inertia=inertia)

    joints = []
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        jtype = joint_el.get("type")
        parent = joint_el.find("parent").get("link")
        child = joint_el.find("child").get("link")
        origin = joint_el.find("origin")
        xyz = np.array(_floats(origin.get("xyz", "0 0 0"))) if origin is not None else np.zeros(3)
        rpy = np.array(_floats(origin.get("rpy", "0 0 0"))) if origin is not None else np.zeros(3)
        axis_el = joint_el.find("axis")
        axis = np.array(_floats(axis_el.get("xyz"))) if axis_el is not None else np.array([1.0, 0.0, 0.0])
        joints.append(dict(name=name, type=jtype, parent=parent, child=child,
                            xyz=xyz, rpy=rpy, axis=axis))

    children_names = {j["child"] for j in joints}
    root_candidates = [name for name in links if name not in children_names]
    assert len(root_candidates) == 1, f"expected exactly one root link, found {root_candidates}"
    root_link = root_candidates[0]

    joints_by_parent = {}
    for j in joints:
        joints_by_parent.setdefault(j["parent"], []).append(j)

    return links, joints_by_parent, root_link


def rpy_to_matrix(rpy):
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)
    cy, sy = jnp.cos(y), jnp.sin(y)
    rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    """Rodrigues' rotation formula, for a revolute joint about an arbitrary axis."""
    axis = axis / jnp.linalg.norm(axis)
    k = jnp.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    return jnp.eye(3) + jnp.sin(angle) * k + (1.0 - jnp.cos(angle)) * (k @ k)


class RigidBodyModel:
    """
    General Euler-Lagrange rigid-body dynamics from a real URDF file, for any
    kinematic tree of revolute/continuous/prismatic/fixed joints -- not tied to
    one specific robot's hand-transcribed numbers. M(q), C(q,qd)qd, g(q) are all
    built from jax.grad/jax.jvp on the same forward-kinematics chain, exactly as
    in the Kinova-Gen3-specific version this generalizes (see gen3_dynamics.py
    and Discovery Experiment 61), so the physics is unchanged -- only the
    structure (from any URDF, not one hardcoded robot) and the DOF count/type
    (revolute or prismatic, not only revolute) are new.
    """

    def __init__(self, urdf_path):
        links, joints_by_parent, root_link = parse_urdf(urdf_path)
        self.links = links
        self.root_link = root_link

        self.dof_joints = []
        self.link_parent_joint = {}

        def walk(link_name):
            for j in joints_by_parent.get(link_name, []):
                self.link_parent_joint[j["child"]] = j
                if j["type"] != "fixed":
                    self.dof_joints.append(j)
                walk(j["child"])

        walk(root_link)
        self.n = len(self.dof_joints)
        self.dof_index = {j["name"]: idx for idx, j in enumerate(self.dof_joints)}

        self.link_ancestor_dofs = {root_link: []}

        def collect(link_name, ancestors):
            self.link_ancestor_dofs[link_name] = list(ancestors)
            for j in joints_by_parent.get(link_name, []):
                next_ancestors = ancestors + ([self.dof_index[j["name"]]] if j["type"] != "fixed" else [])
                collect(j["child"], next_ancestors)

        collect(root_link, [])

        self.link_names = list(links.keys())
        self.mass = jnp.array([links[name]["mass"] for name in self.link_names])
        self.com = jnp.array([links[name]["com"] for name in self.link_names])
        self.inertia = jnp.array([links[name]["inertia"] for name in self.link_names])
        self.link_index = {name: i for i, name in enumerate(self.link_names)}

    def forward_kinematics(self, q):
        """
        Returns (pos, rot, joint_axis_world): dicts from link/joint name to its
        world-frame position (3,)/rotation (3,3), and each movable joint's axis
        expressed in world frame (needed for the Jacobian columns below).
        """
        pos = {self.root_link: jnp.zeros(3)}
        rot = {self.root_link: jnp.eye(3)}
        joint_axis_world = {}

        def walk(link_name, p, r):
            for j in self._children_joints(link_name):
                r_offset = rpy_to_matrix(jnp.asarray(j["rpy"]))
                p_child = p + r @ jnp.asarray(j["xyz"])
                r_child = r @ r_offset
                axis_world = r_child @ (jnp.asarray(j["axis"]) / jnp.linalg.norm(jnp.asarray(j["axis"])))

                if j["type"] in ("revolute", "continuous"):
                    joint_axis_world[j["name"]] = axis_world
                    angle = q[self.dof_index[j["name"]]]
                    r_child = r_child @ axis_angle_matrix(jnp.asarray(j["axis"]), angle)
                elif j["type"] == "prismatic":
                    joint_axis_world[j["name"]] = axis_world
                    disp = q[self.dof_index[j["name"]]]
                    p_child = p_child + axis_world * disp

                pos[j["child"]] = p_child
                rot[j["child"]] = r_child
                walk(j["child"], p_child, r_child)

        walk(self.root_link, jnp.zeros(3), jnp.eye(3))
        return pos, rot, joint_axis_world

    def _children_joints(self, link_name):
        return [j for j in self.link_parent_joint.values() if j["parent"] == link_name]

    def com_positions(self, q):
        pos, rot, _ = self.forward_kinematics(q)
        return jnp.stack([pos[name] + rot[name] @ jnp.asarray(self.links[name]["com"])
                           for name in self.link_names])

    def _link_jacobian(self, link_name, pos, rot, joint_axis_world):
        p_link = pos[link_name] + rot[link_name] @ jnp.asarray(self.links[link_name]["com"])
        jv = jnp.zeros((3, self.n))
        jw = jnp.zeros((3, self.n))
        for dof_idx in self.link_ancestor_dofs[link_name]:
            j = self.dof_joints[dof_idx]
            axis_w = joint_axis_world[j["name"]]
            if j["type"] == "prismatic":
                jv = jv.at[:, dof_idx].set(axis_w)
            else:
                p_joint = pos[j["child"]] - rot[j["child"]] @ (
                    axis_angle_matrix(jnp.asarray(j["axis"]), jnp.array(0.0)) @ jnp.zeros(3))
                p_joint = pos[j["parent"]] + rot[j["parent"]] @ rpy_to_matrix(jnp.asarray(j["rpy"])) @ jnp.zeros(3) \
                    if False else self._joint_origin_world(j, pos, rot)
                jv = jv.at[:, dof_idx].set(jnp.cross(axis_w, p_link - p_joint))
                jw = jw.at[:, dof_idx].set(axis_w)
        return jv, jw

    def _joint_origin_world(self, j, pos, rot):
        return pos[j["parent"]] + rot[j["parent"]] @ jnp.asarray(j["xyz"])

    def mass_matrix(self, q):
        pos, rot, joint_axis_world = self.forward_kinematics(q)
        m = jnp.zeros((self.n, self.n))
        for name in self.link_names:
            if self.links[name]["mass"] == 0.0:
                continue
            jv, jw = self._link_jacobian(name, pos, rot, joint_axis_world)
            i_local = jnp.asarray(self.links[name]["inertia"])
            i_world = rot[name] @ i_local @ rot[name].T
            m = m + self.links[name]["mass"] * (jv.T @ jv) + jw.T @ i_world @ jw
        return m

    def potential_energy(self, q):
        com = self.com_positions(q)
        return jnp.sum(self.mass * com[:, 2]) * G

    def kinetic_energy(self, q, qd):
        m = self.mass_matrix(q)
        return 0.5 * qd @ m @ qd

    def total_energy(self, q, qd):
        return self.kinetic_energy(q, qd) + self.potential_energy(q)

    def gravity_forces(self, q):
        return jax.grad(self.potential_energy)(q)

    def bias_forces(self, q, qd):
        mv = lambda qq: self.mass_matrix(qq) @ qd
        mdot_qd = jax.jvp(mv, (q,), (qd,))[1]
        quad = lambda qq: qd @ self.mass_matrix(qq) @ qd
        return mdot_qd - 0.5 * jax.grad(quad)(q)

    def forward_dynamics(self, q, qd, tau):
        m = self.mass_matrix(q)
        rhs = tau - self.bias_forces(q, qd) - self.gravity_forces(q)
        return jnp.linalg.solve(m, rhs)

    def link_position(self, q, link_name):
        pos, rot, _ = self.forward_kinematics(q)
        return pos[link_name]

    def link_jacobian(self, q, link_name):
        pos, rot, joint_axis_world = self.forward_kinematics(q)
        jv = jnp.zeros((3, self.n))
        for dof_idx in self.link_ancestor_dofs[link_name]:
            j = self.dof_joints[dof_idx]
            axis_w = joint_axis_world[j["name"]]
            if j["type"] == "prismatic":
                jv = jv.at[:, dof_idx].set(axis_w)
            else:
                p_joint = self._joint_origin_world(j, pos, rot)
                jv = jv.at[:, dof_idx].set(jnp.cross(axis_w, pos[link_name] - p_joint))
        return jv
