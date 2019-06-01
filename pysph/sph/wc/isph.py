"""
Incompressible SPH
"""
import numpy
from compyle.api import declare
from pysph.sph.scheme import Scheme, add_bool_argument
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation, Group, MultiStageEquations


def get_particle_array_isph(constants=None, **props):
    isph_props = [
        'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'rho0', 'diag', 'rhs',
        'V0', 'V', 'au', 'av', 'aw'
    ]

    # No of particles
    N = len(props['gid'])
    consts = {
        'np': numpy.array([N], dtype=int),
    }

    if constants:
        consts.update(constants)

    pa = get_particle_array(
        additional_props=isph_props, constants=consts, **props
    )
    pa.add_property('ctr', type='int')
    pa.add_property('coeff', stride=100)
    pa.add_property('col_idx', stride=100, type='int')
    pa.add_property('row_idx', stride=100, type='int')
    pa.add_output_arrays(['p'])
    return pa


class ISPHIntegrator(Integrator):
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


class ISPHDFDIIntegrator(Integrator):
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

        self.compute_accelerations(1)

        self.stage3()

        self.update_domain()

        self.do_post_stage(dt, 2)

    def initial_acceleration(self, t, dt):
        pass


class ISPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0, d_u, d_v,
                   d_w, d_u0, d_v0, d_w0, dt, d_rho0, d_rho, d_V):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw,
               d_V0, d_V, dt):
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_V0[d_idx] = d_V[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_u0, d_v0, d_w0,
               d_x0, d_y0, d_z0, dt, d_au, d_av, d_aw):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + 0.5*dt * (d_u[d_idx] + d_u0[d_idx])
        d_y[d_idx] = d_y0[d_idx] + 0.5*dt * (d_v[d_idx] + d_v0[d_idx])
        d_z[d_idx] = d_z0[d_idx] + 0.5*dt * (d_w[d_idx] + d_w0[d_idx])


class ISPHDIStep(ISPHStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]


class ISPHDFDIStep(ISPHStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw, dt,
               d_V0, d_V):
        # Intermediate step, d_au/v/w does not contain grad p.
        dtb2 = 0.5*dt
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_V0[d_idx] = d_V[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av,
               d_aw, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]


class MomentumEquationBodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationBodyForce, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class VelocityDivergence(Equation):
    def initialize(self, d_idx, d_rhs):
        d_rhs[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_rhs, dt, VIJ, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        vdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_rhs[d_idx] += -Vj * vdotdwij / dt


class VelocityDivergenceDFDI(Equation):
    def initialize(self, d_idx, d_rhs):
        d_rhs[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_rhs, dt, VIJ, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        vdotdwij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_rhs[d_idx] += -2*Vj * vdotdwij / dt


class DensityInvariance(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(DensityInvariance, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_rho, d_rhs, dt):
        rho0 = self.rho0
        d_rhs[d_idx] = (rho0 - d_rho[d_idx]) / (dt*dt*rho0)


class DensityInvarianceDFDI(Equation):
    def post_loop(self, d_idx, d_V, d_V0, d_rhs, dt):
        V0 = d_V0[d_idx]
        d_rhs[d_idx] = 2*(V0 - d_V[d_idx]) / (dt*dt*V0)


class PressureCoeffMatrix(Equation):
    def initialize(self, d_idx, d_ctr, d_diag, d_col_idx):
        # Make only the diagonals zero as the rest are not summed.
        d_diag[d_idx] = 0.0
        d_ctr[d_idx] = 0

        # Initialize col_idx to -1 so as to use it in cond while constructing
        # pressure coeff matrix.
        i = declare('int')
        for i in range(100):
            d_col_idx[d_idx*100 + i] = -1

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_gid, d_coeff, d_ctr,
             d_col_idx, d_row_idx, d_diag, XIJ, DWIJ, R2IJ, EPS):
        rhoij = (s_rho[s_idx] + d_rho[d_idx])
        rhoij2_1 = 1.0/(rhoij*rhoij)

        xdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        fac = 8.0 * s_m[s_idx] * rhoij2_1 * xdotdwij / (R2IJ + EPS)

        j, k = declare('int', 3)
        j = d_gid[s_idx]

        d_diag[d_idx] += fac

        k = int(d_ctr[d_idx])
        d_coeff[d_idx*100 + k] = -fac
        d_col_idx[d_idx*100 + k] = j
        d_row_idx[d_idx*100 + k] = d_idx
        d_ctr[d_idx] += 1


class PPESolve(Equation):
    def py_initialize(self, dst, t, dt):
        import scipy.sparse as sp
        from scipy.sparse.linalg import bicgstab

        coeff = declare('object')
        cond = declare('object')
        n = declare('int')
        n = dst.np[0]

        # Mask all indices which are not used in the construction.
        cond = (dst.col_idx != -1)

        coeff = sp.csr_matrix(
            (dst.coeff[cond], (dst.col_idx[cond], dst.row_idx[cond])),
            shape=(n, n)
        )
        # Add tiny random noise so the matrix is not singular.
        dst.diag -= numpy.random.random(n)

        coeff += sp.diags(dst.diag)

        # Pseudo-Neumann boundary conditions
        dst.rhs[:] -= dst.rhs.mean()

        dst.p[:], ec = bicgstab(coeff, dst.rhs, x0=dst.p)
        assert ec == 0, "Not converging!"


class MomentumEquationPressureGradient(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_rho, s_rho, d_au, d_av, d_aw,
             DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        pij = (d_p[d_idx] - s_p[s_idx])
        fac = Vj * pij / d_rho[d_idx]

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]


class MomentumEquationPressureGradientSymmetric(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_p, s_p, d_rho, s_rho, d_au, d_av, d_aw,
             DWIJ):
        rhoi2 = d_rho[d_idx]*d_rho[d_idx]
        rhoj2 = s_rho[s_idx]*s_rho[s_idx]
        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2
        fac = -s_m[s_idx] * pij

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]


class UpdatePosition(Equation):
    def post_loop(self, d_idx, d_au, d_av, d_aw, d_x, d_y, d_z, dt):
        d_x[d_idx] += d_au[d_idx] * dt*dt * 0.5
        d_y[d_idx] += d_av[d_idx] * dt*dt * 0.5
        d_z[d_idx] += d_aw[d_idx] * dt*dt * 0.5


class CheckDensityError(Equation):
    def __init__(self, dest, sources, rho0, tol=0.01):
        self.conv = 0
        self.rho0 = rho0
        self.tol = tol
        self.count = 0
        self.rho_err = 0
        super(CheckDensityError, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        self.rho_err = numpy.abs(dst.rho - self.rho0).max()
        self.conv = 1 if self.rho_err < self.tol else -1
        self.count += 1

    def converged(self):
        return self.conv


class ISPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, nu, rho0, c0, alpha, beta=0.0,
                 gx=0.0, gy=0.0, gz=0.0, tolerance=0.01, variant="DF",
                 symmetric=False,):
        self.fluids = fluids
        self.solver = None
        self.dim = dim
        self.nu = nu
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.tolerance = tolerance
        self.rho0 = rho0
        self.symmetric = symmetric

    def add_user_options(self, group):
        group.add_argument(
            "--variant", action="store", dest="variant",
            type=str, choices=['DF', 'DI', 'DFDI'],
            help="ISPH variant (defaults to \"DF\" Cummins and Rudmann)."
        )
        group.add_argument(
            '--alpha', action='store', type=float, dest='alpha',
            default=None,
            help='Artificial viscosity.'
        )
        add_bool_argument(
            group, 'symmetric', dest='symmetric', default=None,
            help='Use symmetric form of pressure gradient.'
        )

    def consume_user_options(self, options):
        _vars = ['variant', 'alpha', 'symmetric']
        data = dict((var, self._smart_getattr(options, var))
                    for var in _vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        import pysph.base.kernels as kern
        if kernel is None:
            kernel = kern.QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        if self.variant == "DF":
            step_cls = ISPHStep
        elif self.variant == "DI":
            step_cls = ISPHDIStep
        else:
            step_cls = ISPHDFDIStep

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        if integrator_cls is not None:
            cls = integrator_cls
        elif self.variant == "DFDI":
            cls = ISPHDFDIIntegrator
        else:
            cls = ISPHIntegrator

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def _get_viscous_eqns(self, variant):
        from pysph.sph.wc.transport_velocity import (
            MomentumEquationViscosity, MomentumEquationArtificialViscosity,
            SummationDensity)
        from pysph.sph.wc.viscosity import LaminarViscosity

        all = self.fluids
        eq, stg = [], []
        if variant.endswith('DI'):
            for fluid in self.fluids:
                eq.append(SummationDensity(dest=fluid, sources=all))
            stg.append(Group(equations=eq, real=False))

        eq = []
        for fluid in self.fluids:
            if variant.endswith('DI'):
                eq.append(
                    MomentumEquationViscosity(dest=fluid, sources=all,
                                              nu=self.nu)
                )
            else:
                eq.append(
                    LaminarViscosity(dest=fluid, sources=self.fluids,
                                     nu=self.nu)
                )
            eq.append(
                MomentumEquationArtificialViscosity(
                    dest=fluid, sources=self.fluids, c0=self.c0,
                    alpha=self.alpha
                )
            )
            eq.append(
                MomentumEquationBodyForce(
                    dest=fluid, sources=self.fluids, gx=self.gx, gy=self.gy,
                    gz=self.gz)
            )
        stg.append(Group(equations=eq))
        return stg

    def _get_ppe(self, variant):
        from pysph.sph.wc.transport_velocity import SummationDensity

        all = self.fluids
        eq, stg = [], []
        if variant.endswith('DI'):
            for fluid in self.fluids:
                eq.append(SummationDensity(dest=fluid, sources=all))
            stg.append(Group(equations=eq, real=False))

        eq2 = []
        for fluid in self.fluids:
            if variant == 'DI':
                eq2.append(
                    DensityInvariance(dest=fluid, sources=None, rho0=self.rho0)
                )
            elif variant == 'DFDI':
                eq2.append(
                    DensityInvarianceDFDI(dest=fluid, sources=None)
                )
            else:
                eq2.append(VelocityDivergence(dest=fluid, sources=all))
            eq2.append(PressureCoeffMatrix(dest=fluid, sources=all))
        stg.append(Group(equations=eq2))

        eq22 = []
        for fluid in self.fluids:
            eq22.append(PPESolve(dest=fluid, sources=all))
        stg.append(Group(equations=eq22))
        return stg

    def get_equations(self):
        all = self.fluids

        all_eqns = []
        # Compute Viscous and Body forces
        stg1 = self._get_viscous_eqns(self.variant)
        all_eqns.append(stg1)

        # Converge till particle distribution is good.
        if self.variant == 'DFDI':
            solver_eqns = []
            for fluid in self.fluids:
                solver_eqns.append(Group(equations=[
                    CheckDensityError(dest=fluid, sources=all, rho0=self.rho0,
                                      tol=self.tolerance)
                ]))
            solver_eqns.extend(self._get_ppe(self.variant))

            eq24 = []
            for fluid in self.fluids:
                eq24.append(
                    MomentumEquationPressureGradient(dest=fluid, sources=all)
                )
                eq24.append(
                    UpdatePosition(dest=fluid, sources=None)
                )
            solver_eqns.append(Group(equations=eq24))

            stg_dfdi = [Group(equations=solver_eqns, iterate=True,
                              max_iterations=500, min_iterations=1)]

            stg_dfdi.extend(self._get_viscous_eqns(self.variant))
            all_eqns.append(stg_dfdi)

        # Setup PPE
        variant = 'DF' if self.variant.startswith('DF') else 'DI'
        stg2 = self._get_ppe(variant)

        # Compute acceleration due to pressure, initialize au/av/aw to 0.
        eq4 = []
        for fluid in self.fluids:
            if self.symmetric:
                eq4.append(
                    MomentumEquationPressureGradientSymmetric(dest=fluid,
                                                              sources=all)
                )
            else:
                eq4.append(
                    MomentumEquationPressureGradient(dest=fluid, sources=all)
                )
        stg2.append(Group(equations=eq4))
        all_eqns.append(stg2)

        return MultiStageEquations(all_eqns)

    def setup_properties(self, particles, clean=True):
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_isph(name='junk',
                                        gid=particle_arrays['fluid'].gid)
        props = []
        for x, arr in dummy.properties.items():
            tmp = dict(name=x, type=arr.get_c_type())
            if x in dummy.stride:
                tmp.update(stride=dummy.stride[x])
            props.append(tmp)
        constants = [dict(name=x, data=v) for x, v in dummy.constants.items()]
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            for const in constants:
                pa.add_constant(**const)
