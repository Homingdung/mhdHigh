# Andrews-PhD-2025
# potential form with 3 AVs

from firedrake import *
import csv
from mpi4py import MPI
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
import os
from ufl.algorithms.ad import expand_derivatives

# space paramters
baseN = 3

# physical parameters
Re = Constant(1)
Rem = Constant(1)
S = Constant(1)
RH = Constant(1)

# time parameters
stage = 1
t = Constant(0)
dt = Constant(0.01)
T = 1.0

def energy(u, A):
    return assemble(0.5 * inner(u, u) * dx) + assemble(S * inner(curl(A), curl(A)) * dx)

def helicity(A):
    return assemble(inner(A, curl(A)) * dx)


mesh = PeriodicUnitCubeMesh(baseN, baseN, baseN)
(x, y, z0) = SpatialCoordinate(mesh)

Vg = FunctionSpace(mesh, "CG", 1)
Vc = FunctionSpace(mesh, "N1curl", 1)

Z = MixedFunctionSpace([Vc, Vc, Vc, Vc, Vc, Vg, Vg])
z = Function(Z)
z_test = TestFunction(Z)

(u, A, j, H, w, p, phi) = split(z)
(ut, At, jt, Ht, wt, pt, phit) = split(z_test)

butcher_tableau=GaussLegendre(stage)


# initial condition, Mao-Xi-2025
def g(x):
    return 32 * x**3 * (x - 1) ** 3

A_ex = as_vector([y*g(x) * g(y) * g(z0), -x*g(x)*g(y)*g(z0), g(x)*g(y)*g(z0)])

z.sub(1).interpolate(A_ex)


F = (
    # equation for u
      inner(Dt(u), ut) * dx
    - inner(cross(u, w), ut) * dx
    + inner(grad(p), ut) * dx
    - S * inner(cross(j, H), ut) * dx
    + 1/Re * inner(curl(u), curl(ut)) * dx
    # equation for A
    + inner(Dt(A), At) * dx
    - inner(cross(u, H), At) * dx
    + inner(grad(phi), At) * dx
    + RH * inner(cross(j, H), At) * dx
    + 1/Rem * inner(j, At) * dx
    # equation for j
    + inner(j, jt) * dx
    - inner(curl(A), curl(jt)) * dx
    # equaiton for H
    + inner(H, Ht) * dx
    - inner(curl(A), Ht) * dx
    # equation for w
    + inner(w, wt) * dx
    - inner(curl(u), wt) * dx
    # LM 1
    + inner(u, grad(pt)) * dx
    # LM 2 
    + inner(A, grad(phit)) * dx
)

lu = {
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

sp = lu

stepper = TimeStepper(F, butcher_tableau, t, dt, z, solver_parameters = sp)

while (float(t) < float(T - dt) + 1.0e-10):
    t.assign(t + dt)
    if mesh.comm.rank == 0:
        print(GREEN % f"Solving for t = {float(t):.4f}")
    stepper.advance()


