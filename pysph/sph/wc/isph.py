"""
Incompressible SPH
"""
import numpy as np
from compyle.api import declare
from pysph.sph.scheme import Scheme
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation, Group, MultiStageEquations


def get_particle_array_isph(constants=None, **props):
    isph_props = [
        'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'rho0'
    ]

    # No of particles
    nop = len(props['gid'])
    consts = {
        'nop': np.array([nop], dtype=int),
        'lhs': np.zeros([nop*nop], dtype=float),
        'rhs': np.zeros([nop], dtype=float),
    }

    if constants:
        consts.update(constants)

    pa = get_particle_array(
        additional_props=isph_props, constants=consts, **props
    )
    pa.add_output_arrays(['p'])
    return pa


class PECIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations(0)

        self.stage1()

        self.update_domain()

        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations(1)

        self.stage2()

        self.update_domain()

        self.do_post_stage(dt, 2)

    def initial_acceleration(self, t, dt):
        pass


class ISPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v,
                   d_w, d_u0, d_v0, d_w0, dt):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw,
               dt):
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_u0, d_v0, d_w0,
               d_x0, d_y0, d_z0, dt, d_au, d_av, d_aw):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + 0.5*dt * (d_u[d_idx] + d_u0[d_idx])
        d_y[d_idx] = d_y0[d_idx] + 0.5*dt * (d_v[d_idx] + d_v0[d_idx])
        d_z[d_idx] = d_z0[d_idx] + 0.5*dt * (d_w[d_idx] + d_w0[d_idx])


class MomentumEquationViscosity(Equation):
    def __init__(self, dest, sources, nu, gx=0.0, gy=0.0, gz=0.0):
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_idx, s_idx, s_m, d_rho, d_au, d_av, d_aw, XIJ, DWIJ, R2IJ,
             EPS, VIJ):
        nu = self.nu
        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
        fac = 2.0 * s_m[s_idx] * nu * xdotdwij / (R2IJ + EPS)

        d_au[d_idx] += fac * VIJ[0]
        d_av[d_idx] += fac * VIJ[1]
        d_aw[d_idx] += fac * VIJ[2]


class VelocityDivergence(Equation):
    def initialize(self, d_idx, d_rhs):
        d_rhs[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_rhs, dt, VIJ, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        vdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_rhs[d_idx] += -Vj * vdotdwij / dt


class PressureCoeffMatrix(Equation):
    def initialize(self, d_idx, d_nop, d_lhs, d_gid):
        # Make only the diagonals zero as the rest are not summed.
        d_lhs[d_idx*d_nop[0] + d_gid[d_idx]] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_nop, d_lhs, d_gid, XIJ, DWIJ,
             R2IJ, EPS):
        rhoj1 = 1.0/s_rho[s_idx]
        Vj = s_m[s_idx] / s_rho[s_idx]
        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
        fac = 2.0 * Vj * rhoj1 * xdotdwij / (R2IJ + EPS)

        n, j = declare('int', 2)
        n = d_nop[0]
        j = d_gid[s_idx]
        if d_idx != j:
            d_lhs[d_idx*n + j] = -fac
        d_lhs[d_idx*n + d_idx] += fac


class PPESolve(Equation):
    # def py_initialize(self, dst, t, dt):
    def reduce(self, dst, t, dt):
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import bicgstab

        coeff = declare('object')
        # precond = declare('object')
        n = declare('int')
        n = dst.nop[0]
        coeff = dst.lhs.reshape(n, n)

        # import pudb; pudb.set_trace()
        # Pseudo-Neumann boundary conditions
        dst.rhs[:] -= np.mean(dst.rhs)

        # Set coeff of 1st particle to zero
        coeff -= np.diag(np.random.random(n))

        # precond = np.diag(np.diag(coeff))
        coeff = sp.csr_matrix(coeff)

        # Use of precond makes it slow.
        # dst.p[:], exitcode = bicgstab(coeff, dst.rhs, x0=dst.p, M=precond)
        dst.p[:], exitcode = bicgstab(coeff, dst.rhs, x0=dst.p)


class MomentumEquationPressureGradient(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_rho, s_rho, d_au, d_av, d_aw,
             DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        pji = (s_p[s_idx] - d_p[d_idx])
        fac = -Vj * pji / d_rho[d_idx]

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]


class FreeSurface(Equation):
    def __init__(self, dest, sources, beta):
        self.beta = beta
        super(FreeSurface, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_divr, Vj, XDOTDWIJ):
        d_divr += -Vj * XDOTDWIJ

    def post_loop(self, d_divr, d_gid, d_idx, d_nop, A):
        if (d_divr[d_idx] < self.beta):
            A[d_idx * d_nop[0] * d_gid[d_idx]] = 1.0


class MESchwaiger(MomentumEquationViscosity):
    def initialize(self, d_idx, d_au, d_av, d_aw, d_fx, d_fy, d_fz):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

        d_fx[d_idx] = 0
        d_fy[d_idx] = 0
        d_fz[d_idx] = 0

    def loop(self, d_idx, s_idx, s_m, d_au, d_av, d_aw, d_fx, d_fy, d_fz,
             RHOIJ1, XIJ, DWIJ, VIJ, R2IJ, EPS, XDOTDWIJ):
        nu = self.viscous_factor * RHOIJ1
        vij = s_m[s_idx] * RHOIJ1
        fac = 2.0 * vij * nu * XDOTDWIJ / (R2IJ + EPS)

        d_au[d_idx] += fac * VIJ[0]
        d_av[d_idx] += fac * VIJ[1]
        d_aw[d_idx] += fac * VIJ[2]

        fac = 2.0 * nu * vij
        d_fx[d_idx] += fac * DWIJ[0]
        d_fy[d_idx] += fac * DWIJ[1]
        d_fz[d_idx] += fac * DWIJ[2]

    def post_loop(self, d_au, d_av, d_aw, d_fx, d_fy, d_fz, d_idx, d_v00,
                  d_v01, d_v10, d_v11, d_v02, d_v12, d_v20, d_v21, d_v22):
        d_au[d_idx] += (d_fx[d_idx]*d_v00[d_idx] + d_fy[d_idx]*d_v01[d_idx]
                        + d_fz[d_idx]*d_v02[d_idx])
        d_av[d_idx] += (d_fx[d_idx]*d_v10[d_idx] + d_fy[d_idx]*d_v11[d_idx]
                        + d_fz[d_idx] * d_v12[d_idx])
        d_aw[d_idx] += (d_fx[d_idx]*d_v00[d_idx] + d_fy[d_idx]*d_v01[d_idx]
                        + d_fz[d_idx] * d_v22[d_idx])


class UpdateVelandPos(Equation):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v, d_w,
                   d_u0, d_v0, d_w0, d_au, d_av, d_aw, d_rho, d_rho0, dt):

        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]


class DensityInvariance(Equation):
    def initialize(self, d_idx, d_rhs, d_rho0, d_rho, dt):
        d_rhs[d_idx] = (d_rho0[d_idx] - d_rho[d_idx]) / (d_rho0[d_idx]*dt*dt)


class ISPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, nu, gx=0.0, gy=0.0, gz=0.0,
                 variant="CR"):
        self.fluids = fluids
        self.solver = None
        self.dim = dim
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.variant = variant

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        import pysph.base.kernels as kern
        if kernel is None:
            kernel = kern.QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = ISPHStep
        if self.variant == "SL":
            # TODO: Implement this
            step_cls = None

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        if integrator_cls is not None:
            cls = integrator_cls
        else:
            cls = PECIntegrator

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        all = self.fluids

        eq1, stg1 = [], []
        for fluid in self.fluids:
            eq1.append(
                MomentumEquationViscosity(
                    dest=fluid, sources=all, nu=self.nu, gx=self.gx,
                    gy=self.gy, gz=self.gz)
            )
        stg1.append(Group(equations=eq1))

        eq2, stg2 = [], []
        for fluid in self.fluids:
            eq2.append(PressureCoeffMatrix(dest=fluid, sources=all))
            eq2.append(VelocityDivergence(dest=fluid, sources=all))
        stg2.append(Group(equations=eq2))

        eq3 = []
        for fluid in self.fluids:
            eq3.append(PPESolve(dest=fluid, sources=all))
        stg2.append(Group(equations=eq3))

        eq4 = []
        for fluid in self.fluids:
            eq4.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all)
            )
        stg2.append(Group(equations=eq4))

        return MultiStageEquations([stg1, stg2])

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_isph(name='junk',
                                        gid=particle_arrays['fluid'].gid)
        props = list(dummy.properties.keys())
        props += [dict(name=x, stride=v) for x, v in dummy.stride.items()]
        constants = [dict(name=x, data=v) for x, v in dummy.constants.items()]
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            for const in constants:
                pa.add_constant(**const)
