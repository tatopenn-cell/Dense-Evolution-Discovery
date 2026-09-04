import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Kinova Gen3 7-DoF, real link masses/inertias/joint origins, taken directly
# from GEN3_URDF_V12.urdf (github.com/vincekurtz/passivity_cbf_demo,
# models/gen3_7dof/urdf/) -- the same URDF Kurtz, Wensing & Lin (2021,
# arXiv:2109.13349) load in Drake for their CBF-passivity controller.
# Values not derived or invented here, only copied and reorganized.

N = 7
G = 9.81

JOINT_XYZ = jnp.array([
    [0.0, 0.0, 0.15643],
    [0.0, 0.005375, -0.12838],
    [0.0, -0.21038, -0.006375],
    [0.0, 0.006375, -0.21038],
    [0.0, -0.20843, -0.006375],
    [0.0, 0.00017505, -0.10593],
    [0.0, -0.10593, -0.00017505],
])

JOINT_RPY = jnp.array([
    [3.1416, 2.7629e-18, -4.9305e-36],
    [1.5708, 2.1343e-17, -1.1102e-16],
    [-1.5708, 1.2326e-32, -2.9122e-16],
    [1.5708, -6.6954e-17, -1.6653e-16],
    [-1.5708, 2.2204e-16, -6.373e-17],
    [1.5708, 9.2076e-28, -8.2157e-15],
    [-1.5708, -5.5511e-17, 9.6396e-17],
])

LINK_MASS = jnp.array([1.3773, 1.1636, 1.1636, 0.9302, 0.6781, 0.6781, 0.5006])

LINK_COM = jnp.array([
    [-2.3e-05, -0.010364, -0.07336],
    [-4.4e-05, -0.09958, -0.013278],
    [-4.4e-05, -0.006641, -0.117892],
    [-1.8e-05, -0.075478, -0.015006],
    [1e-06, -0.009432, -0.063883],
    [1e-06, -0.045483, -0.00965],
    [-0.000281, -0.011402, -0.029798],
])


def _sym_inertia(ixx, ixy, ixz, iyy, iyz, izz):
    return jnp.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


LINK_INERTIA = jnp.stack([
    _sym_inertia(0.00457, 1e-06, 2e-06, 0.004831, 0.000448, 0.001409),
    _sym_inertia(0.011088, 5e-06, 0.0, 0.001072, -0.000691, 0.011255),
    _sym_inertia(0.010932, 0.0, -7e-06, 0.011127, 0.000606, 0.001043),
    _sym_inertia(0.008147, -1e-06, 0.0, 0.000631, -0.0005, 0.008316),
    _sym_inertia(0.001596, 0.0, 0.0, 0.001607, 0.000256, 0.000399),
    _sym_inertia(0.001641, 0.0, 0.0, 0.00041, -0.000278, 0.001641),
    _sym_inertia(0.000587, 3e-06, 3e-06, 0.000369, 0.000118, 0.000609),
])


def rpy_to_matrix(rpy):
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)
    cy, sy = jnp.cos(y), jnp.sin(y)
    rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def rot_z(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def forward_kinematics(q):
    p = jnp.zeros(3)
    r = jnp.eye(3)
    positions, rotations, z_axes = [], [], []
    for i in range(N):
        r_offset = rpy_to_matrix(JOINT_RPY[i])
        p = p + r @ JOINT_XYZ[i]
        r = r @ r_offset
        z_axes.append(r[:, 2])
        r = r @ rot_z(q[i])
        positions.append(p)
        rotations.append(r)
    return positions, rotations, z_axes


def com_positions(q):
    positions, rotations, _ = forward_kinematics(q)
    return jnp.stack([positions[i] + rotations[i] @ LINK_COM[i] for i in range(N)])


def link_jacobians(q):
    positions, rotations, z_axes = forward_kinematics(q)
    com = jnp.stack([positions[i] + rotations[i] @ LINK_COM[i] for i in range(N)])

    jv = jnp.zeros((N, 3, N))
    jw = jnp.zeros((N, 3, N))
    for i in range(N):
        for j in range(i + 1):
            jw = jw.at[i, :, j].set(z_axes[j])
            jv = jv.at[i, :, j].set(jnp.cross(z_axes[j], com[i] - positions[j]))
    return jv, jw, rotations


def mass_matrix(q):
    jv, jw, rotations = link_jacobians(q)
    m = jnp.zeros((N, N))
    for i in range(N):
        i_world = rotations[i] @ LINK_INERTIA[i] @ rotations[i].T
        m = m + LINK_MASS[i] * (jv[i].T @ jv[i]) + jw[i].T @ i_world @ jw[i]
    return m


def potential_energy(q):
    com = com_positions(q)
    return jnp.sum(LINK_MASS * com[:, 2]) * G


def kinetic_energy(q, qd):
    m = mass_matrix(q)
    return 0.5 * qd @ m @ qd


def gravity_forces(q):
    return jax.grad(potential_energy)(q)


def bias_forces(q, qd):
    mv = lambda qq: mass_matrix(qq) @ qd
    mdot_qd = jax.jvp(mv, (q,), (qd,))[1]
    quad = lambda qq: qd @ mass_matrix(qq) @ qd
    return mdot_qd - 0.5 * jax.grad(quad)(q)


def forward_dynamics(q, qd, tau):
    m = mass_matrix(q)
    rhs = tau - bias_forces(q, qd) - gravity_forces(q)
    return jnp.linalg.solve(m, rhs)


def total_energy(q, qd):
    return kinetic_energy(q, qd) + potential_energy(q)


# Fixed EE frame, from the URDF's "EndEffector" joint (bracelet_no_vision_link -> end_effector_link)
EE_XYZ = jnp.array([0.0, 0.0, -0.0615250000000001])
EE_RPY = jnp.array([3.14159265358979, 1.09937075168372e-32, 0.0])


def end_effector_position(q):
    positions, rotations, _ = forward_kinematics(q)
    return positions[-1] + rotations[-1] @ EE_XYZ


def end_effector_jacobian(q):
    return jax.jacfwd(end_effector_position)(q)


def end_effector_jacobian_dot_times_qd(q, qd):
    jac_v = lambda qq: end_effector_jacobian(qq) @ qd
    return jax.jvp(jac_v, (q,), (qd,))[1]
