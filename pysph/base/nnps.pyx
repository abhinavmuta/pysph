# numpy import
import numpy as np
cimport numpy as np

# malloc and friends
from libc.stdlib cimport malloc, free

# cpython
from cpython.dict cimport PyDict_Clear, PyDict_Contains, PyDict_GetItem
from cpython.list cimport PyList_GetItem

# Cython for compiler directives
cimport cython

cdef extern from 'math.h':
    int abs(int)
    double ceil(double)
    double floor(double)
    double fabs(double)
    double fmax(double, double)
    double fmin(double, double)

cdef extern from 'limits.h':
    cdef unsigned int UINT_MAX
    cdef int INT_MAX

# ZOLTAN ID TYPE AND PTR
ctypedef unsigned int ZOLTAN_ID_TYPE
ctypedef unsigned int* ZOLTAN_ID_PTR

# Particle Tag information
from utils import ParticleTAGS

cdef int Local = ParticleTAGS.Local
cdef int Remote = ParticleTAGS.Remote
cdef int Ghost = ParticleTAGS.Ghost

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int flatten(cIntPoint cid, IntArray ncells, int dim):
    """Return a flattened index for a cell

    The flattening is determined using the row-order indexing commonly
    employed in SPH. This would need to be changed for hash functions
    based on alternate orderings.

    """
    cdef int ncx = ncells.data[0]
    cdef int ncy = ncells.data[1]

    return <int>( cid.x + ncx * cid.y + ncx*ncy * cid.z )

def py_flatten(IntPoint cid, IntArray ncells, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = cid.data
    cdef int flattened_index = flatten( _cid, ncells, dim )
    return flattened_index

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cIntPoint unflatten(int cell_index, IntArray ncells, int dim):
    """Un-flatten a linear cell index"""
    cdef int ncx = ncells.data[0]
    cdef int ncy = ncells.data[1]

    cdef cIntPoint cid
    cdef int ix=0, iy=0, iz=0, tmp=0

    if dim > 1:    
        if dim > 2:
            tmp = ncx * ncy

            iz = cell_index/tmp
            cell_index = cell_index - iz * tmp


        iy = cell_index/ncx
        ix = cell_index - (iy * ncx)

    else:
        ix = cell_index
    
    # return the tuple cell index
    cid = cIntPoint_new( ix, iy, iz )
    return cid

def py_unflatten(int cell_index, IntArray ncells, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = unflatten( cell_index, ncells, dim )
    cdef IntPoint cid = IntPoint_from_cIntPoint(_cid)
    return cid

cdef inline int real_to_int(double real_val, double step):
    """ Return the bin index to which the given position belongs.

    Parameters:
    -----------
    val -- The coordinate location to bin
    step -- the bin size

    Example:
    --------
    real_val = 1.5, step = 1.0 --> ret_val = 1

    real_val = -0.5, step = 1.0 --> real_val = -1

    """
    cdef int ret_val = <int>floor( real_val/step )

    return ret_val

cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size):
    """ Find the cell index for the corresponding point

    Parameters:
    -----------
    pnt -- the point for which the index is sought
    cell_size -- the cell size to use
    id -- output parameter holding the cell index

    Algorithm:
    ----------
    performs a box sort based on the point and cell size

    Notes:
    ------
    Uses the function  `real_to_int`

    """
    cdef cIntPoint p = cIntPoint(real_to_int(pnt.x, cell_size),
                                 real_to_int(pnt.y, cell_size),
                                 real_to_int(pnt.z, cell_size))
    return p

cdef inline cPoint _get_centroid(double cell_size, cIntPoint cid):
    """ Get the centroid of the cell.

    Parameters:
    -----------

    cell_size : double (input)
    Cell size used for binning

    cid : cPoint (input)
    Spatial index for a cell

    Returns:
    ---------

    centroid : cPoint

    Notes:
    ------
    The centroid in any coordinate direction is defined to be the
    origin plus half the cell size in that direction

    """
    centroid = cPoint_new(0.0, 0.0, 0.0)
    centroid.x = (<double>cid.x + 0.5)*cell_size
    centroid.y = (<double>cid.y + 0.5)*cell_size
    centroid.z = (<double>cid.z + 0.5)*cell_size

    return centroid

def get_centroid(double cell_size, IntPoint cid):
    """ Get the centroid of the cell.

    Parameters:
    -----------

    cell_size : double (input)
        Cell size used for binning

    cid : IntPoint (input)
        Spatial index for a cell

    Returns:
    ---------

    centroid : Point

    Notes:
    ------
    The centroid in any coordinate direction is defined to be the
    origin plus half the cell size in that direction

    """
    cdef cPoint _centroid = _get_centroid(cell_size, cid.data)
    centroid = Point_new(0.0, 0.0, 0.0)

    centroid.data = _centroid
    return centroid

cpdef UIntArray arange_uint(int start, int stop=-1):
    """Utility function to return a numpy.arange for a UIntArray"""
    cdef int size
    cdef UIntArray arange
    cdef int i = 0

    if stop == -1:
        arange = UIntArray(start)
        for i in range(start):
            arange.data[i] = <unsigned int>i
    else:
        size = stop-start
        arange = UIntArray(size)
        for i in range(size):
            arange.data[i] = <unsigned int>(start + i)

    return arange

#################################################################
# NNPS extension classes
#################################################################
cdef class NNPSParticleArrayWrapper:
    def __init__(self, ParticleArray pa):
        self.pa = pa
        self.name = pa.name

        self.x = pa.get_carray('x')
        self.y = pa.get_carray('y')
        self.z = pa.get_carray('z')
        self.h = pa.get_carray('h')

        self.gid = pa.get_carray('gid')
        self.tag = pa.get_carray('tag')

        self.np = pa.get_number_of_particles()

    cdef int get_number_of_particles(self):
        cdef ParticleArray pa = self.pa
        return pa.get_number_of_particles()

cdef class DomainLimits:
    """This class determines the limits of the solution domain.

    We expect all simulations to have well defined domain limits
    beyond which we are either not interested or the solution is
    invalid to begin with. Thus, if a particle leaves the domain,
    the solution should be considered invalid (at least locally).

    The initial domain limits could be given explicitly or asked to be
    computed from the particle arrays. The domain could be periodic.

    """
    def __init__(self, int dim=1, double xmin=-1000, double xmax=1000, double ymin=0,
                 double ymax=0, double zmin=0, double zmax=0,
                 periodic_in_x=False, periodic_in_y=False, periodic_in_z=False):
        """Constructor"""
        self._check_limits(dim, xmin, xmax, ymin, ymax, zmin, zmax)

        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax

        # Indicates if the domain is periodic
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z
        self.is_periodic = periodic_in_x or periodic_in_y or periodic_in_z

        # get the translates in each coordinate direction
        self.xtranslate = xmax - xmin
        self.ytranslate = ymax - ymin
        self.ztranslate = zmax - zmin

        # store the dimension
        self.dim = dim

    def _check_limits(self, dim, xmin, xmax, ymin, ymax, zmin, zmax):
        """Sanity check on the limits"""
        if ( (xmax < xmin) or (ymax < ymin) or (zmax < zmin) ):
            raise ValueError("Invalid domain limits!")

cdef class Cell:
    """Basic indexing structure for the box-sort NNPS.

    For a spatial indexing based on the box-sort algorithm, this class
    defines the spatial data structure used to hold particle indices
    (local and global) that are within this cell.

    """
    def __init__(self, IntPoint cid, double cell_size, int narrays,
                 int layers=2):
        """Constructor

        Parameters:
        -----------

        cid : IntPoint
            Spatial index (unflattened) for the cell

        cell_size : double
            Spatial extent of the cell in each dimension

        narrays : int
            Number of arrays being binned

        layers : int
            Factor to compute the bounding box

        """
        self._cid = cIntPoint_new(cid.x, cid.y, cid.z)
        self.cell_size = cell_size

        self.narrays = narrays

        self.lindices = [UIntArray() for i in range(narrays)]
        self.gindices = [UIntArray() for i in range(narrays)]

        self.nparticles = [lindices.length for lindices in self.lindices]
        self.is_boundary = False

        # compute the centroid for the cell
        self.centroid = _get_centroid(cell_size, cid.data)

        # cell bounding box
        self.layers = layers
        self._compute_bounding_box(cell_size, layers)

        # list of neighboring processors
        self.nbrprocs = IntArray(0)

        # current size of the cell
        self.size = 0

    cpdef set_indices(self, int index, UIntArray lindices, UIntArray gindices):
        """Set the global and local indices for the cell"""
        self.lindices[index] = lindices
        self.gindices[index] = gindices
        self.nparticles[index] = lindices.length

    def get_centroid(self, Point pnt):
        """Utility function to get the centroid of the cell.

        Parameters:
        -----------

        pnt : Point (input/output)
            The centroid is cmoputed and stored in this object.

        The centroid is defined as the origin plus half the cell size
        in each dimension.

        """
        cdef cPoint centroid = self.centroid
        pnt.data = centroid

    def get_bounding_box(self, Point boxmin, Point boxmax, int layers = 1,
                         cell_size=None):
        """Compute the bounding box for the cell.

        Parameters:
        ------------

        boxmin : Point (output)
            The bounding box min coordinates are stored here

        boxmax : Point (output)
            The bounding box max coordinates are stored here

        layers : int (input) default (1)
            Number of offset layers to define the bounding box

        cell_size : double (input) default (None)
            Optional cell size to use to compute the bounding box.
            If not provided, the cell's size will be used.

        """
        if cell_size is None:
            cell_size = self.cell_size

        self._compute_bounding_box(cell_size, layers)
        boxmin.data = self.boxmin
        boxmax.data = self.boxmax

    cdef _compute_bounding_box(self, double cell_size,
                               int layers):
        self.layers = layers
        cdef cPoint centroid = self.centroid
        cdef cPoint boxmin = cPoint_new(0., 0., 0.)
        cdef cPoint boxmax = cPoint_new(0., 0., 0.)

        boxmin.x = centroid.x - (layers+0.5) * cell_size
        boxmax.x = centroid.x + (layers+0.5) * cell_size

        boxmin.y = centroid.y - (layers+0.5) * cell_size
        boxmax.y = centroid.y + (layers+0.5) * cell_size

        boxmin.z = centroid.z - (layers + 0.5) * cell_size
        boxmax.z = centroid.z + (layers + 0.5) * cell_size

        self.boxmin = boxmin
        self.boxmax = boxmax

cdef class NNPS:
    """Nearest neighbor query class using the box-sort algorithm.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint warn=True):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Dimension (Not sure if this is really needed)

        particles : list
            The list of particles we are working on

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainLimits, default (None)
            Optional limits for the domain

        warn : bint
            Flag to warn when extending particle lists

        """
        self.in_parallel = False
        # store the list of particles and number of arrays
        self.particles = particles
        self.narrays = len( particles )

        # create the particle array wrappers
        self.pa_wrappers = [NNPSParticleArrayWrapper(pa) for pa in particles]

        # radius scale and problem dimensionality
        self.radius_scale = radius_scale
        self.dim = dim

        self.domain = domain
        if domain is None:
            self.domain = DomainLimits(dim)

        # periodicity
        self.is_periodic = self.domain.is_periodic

        # warn
        self.warn = warn

    cpdef update(self):
        raise RuntimeError("NNPS :: update called")

    cdef _bin(self, int pa_index, UIntArray indices):
        raise RuntimeError("NNPS :: _bin called")

    cdef _compute_cell_size(self):
        raise RuntimeError("NNPS :: _compute_cell_size called")

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        raise RuntimeError("NNPS :: get_nearest_particles called")

    cpdef brute_force_neighbors(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[src_index]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[dst_index]

        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h

        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h

        cdef double cell_size = self.cell_size
        cdef double radius_scale = self.radius_scale

        cdef size_t num_particles, j

        num_particles = s_x.length
        cdef double xi = d_x.data[d_idx]
        cdef double yi = d_y.data[d_idx]
        cdef double zi = d_z.data[d_idx]

        cdef double hi = d_h.data[d_idx] * radius_scale # gather radius
        cdef double xj, yj, hj, xij2, xij

        # reset the neighbors
        nbrs.reset()

        for j in range(num_particles):
            xj = s_x.data[j]; yj = s_y.data[j]; zj = s_z.data[j];
            hj = radius_scale * s_h.data[j] # scatter radius

            xij2 = (xi - xj)*(xi - xj) + \
                   (yi - yj)*(yi - yj) + \
                   (zi - zj)*(zi - zj)
            xij = sqrt(xij2)
            
            if ( (xij < hi) or (xij < hj) ):
                nbrs.append( <ZOLTAN_ID_TYPE> j )

        return nbrs

    def set_in_parallel(self, bint in_parallel):
        self.in_parallel = in_parallel

    ####################################################################
    # Functions for periodicity
    ####################################################################
    def remove_periodic_ghosts(self):
        """Remove all particles with the Ghost """
        for pa in self.particles:
            pa.remove_tagged_particles(Ghost)

    def adjust_particles(self):
        """Adjust particles for periodicity"""
        cdef DomainLimits domain = self.domain
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray x, y, z

        cdef double xmin = domain.xmin, xmax = domain.xmax
        cdef double ymin = domain.ymin, ymax = domain.ymax,
        cdef double zmin = domain.zmin, zmax = domain.zmax
        cdef double xtranslate = domain.xtranslate
        cdef double ytranslate = domain.ytranslate
        cdef double ztranslate = domain.ztranslate

        cdef bint periodic_in_x = domain.periodic_in_x
        cdef bint periodic_in_y = domain.periodic_in_y
        cdef bint periodic_in_z = domain.periodic_in_z

        cdef double xi, yi, zi

        cdef int i, np

        for pa_wrapper in self.pa_wrappers:
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z

            np = x.length
            for i in range(np):

                if ( (periodic_in_x and not periodic_in_y) ):
                    if x.data[i] < xmin : x.data[i] += xtranslate
                    if x.data[i] > xmax : x.data[i] -= xtranslate

                if ( (periodic_in_y and not periodic_in_x) ):
                    if y.data[i] < ymin : y.data[i] += ytranslate
                    if y.data[i] > ymax : y.data[i] -= ytranslate

                if ( periodic_in_x and periodic_in_y ):
                    if x.data[i] < xmin : x.data[i] += xtranslate
                    if x.data[i] > xmax : x.data[i] -= xtranslate
                    if y.data[i] < ymin : y.data[i] += ytranslate
                    if y.data[i] > ymax : y.data[i] -= ytranslate

    def create_periodic_ghosts(self):
        """Create new ghost particles"""
        cdef DomainLimits domain = self.domain
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef ParticleArray pa
        cdef DoubleArray x, y, z

        cdef double xmin = domain.xmin, xmax = domain.xmax
        cdef double ymin = domain.ymin, ymax = domain.ymax,
        cdef double zmin = domain.zmin, zmax = domain.zmax
        cdef double xtranslate = domain.xtranslate
        cdef double ytranslate = domain.ytranslate
        cdef double ztranslate = domain.ztranslate

        cdef bint periodic_in_x = domain.periodic_in_x
        cdef bint periodic_in_y = domain.periodic_in_y
        cdef bint periodic_in_z = domain.periodic_in_z

        cdef double xi, yi, zi
        cdef double cell_size = self.cell_size

        cdef int i, np, nghost = 0

        # temporary indices for particles to be replicated
        cdef LongArray left, right, top, bottom, lt, rt, lb, rb

        for pa_wrapper in self.pa_wrappers:
            pa = pa_wrapper.pa
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z

            left = LongArray(); right = LongArray()
            top = LongArray(); bottom = LongArray()

            lt = LongArray(); rt = LongArray()
            lb = LongArray(); rb = LongArray()

            np = x.length
            for i in range(np):
                xi = x.data[i]; yi = y.data[i]; zi = z.data[i]

                # X-Periodic
                if periodic_in_x and not periodic_in_y:
                    if ( (xi - xmin) <= cell_size ): left.append(i)
                    if ( (xmax - xi) <= cell_size ): right.append(i)

                # Y-Periodic
                if periodic_in_y and not periodic_in_x:
                    if ( (yi - ymin) <= cell_size ): bottom.append(i)
                    if ( (ymax - yi) <= cell_size ): top.append(i)

                # X&Y-Periodic
                if periodic_in_x and periodic_in_y:
                    # particle near the left end
                    if ( (xi - xmin) <= cell_size ):
                        left.append(i)

                        # left bottom corner
                        if ( (yi - ymin) <= cell_size ):
                            lb.append(i)

                        # left top corner
                        if ( (ymax - yi) < cell_size ):
                            lt.append( i )

                    # particle near the right
                    if ( (xmax - xi) <= cell_size ):
                        right.append(i)

                        # right bottom corner
                        if ( (yi - ymin) <= cell_size ):
                            rb.append(i)

                        # right top corner
                        if ( (ymax - yi) < cell_size ):
                            rt.append( i )

                    # particle near the top
                    if ( (ymax - yi) < cell_size ):
                        top.append(i)

                    # particle near the bottom
                    if ( (yi - ymin) < cell_size ):
                        bottom.append(i)

            # now treat each case separately and count the number of ghosts

            # left
            copy = pa.extract_particles( left )
            copy.x += xtranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # right
            copy = pa.extract_particles( right )
            copy.x -= xtranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # top
            copy = pa.extract_particles( top )
            copy.y -= ytranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # bottom
            copy = pa.extract_particles( bottom )
            copy.y += ytranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # left top
            copy = pa.extract_particles( lt )
            copy.x += xtranslate
            copy.y -= ytranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # left bottom
            copy = pa.extract_particles( lb )
            copy.x += xtranslate
            copy.y += ytranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # right top
            copy = pa.extract_particles( rt )
            copy.x -= xtranslate
            copy.y -= ytranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

            # right bottom
            copy = pa.extract_particles( rb )
            copy.x -= xtranslate
            copy.y += ytranslate
            copy.tag[:] = Ghost
            pa.append_parray(copy)

            nghost += copy.get_number_of_particles()

cdef class BoxSortNNPS(NNPS):
    """Nearest neighbor query class using the box-sort algorithm.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, warn=True):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Dimension (Not sure if this is really needed)

        particles : list
            The list of particles we are working on

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainLimits, default (None)
            Optional limits for the domain

        """
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain, warn)
        
        # default not in parallel

        self.in_parallel = False

        # initialize the cells dict
        self.cells = {}

        # compute the intial box sort
        self.update()

    cpdef update(self):
        """Update the local data after a parallel update.

        We want the NNPS to be independent of the ParallelManager
        which is solely responsible for distributing particles across
        available processors. We assume therefore that after a
        parallel update, each processor has all the local particle
        information it needs.

        """
        cdef dict cells = self.cells
        cdef int i, num_particles
        cdef ParticleArray pa
        cdef UIntArray indices

        # clear the cells dict
        PyDict_Clear( cells )

        # compute the cell size
        self._compute_cell_size()

        # Periodicity is handled by adjusting particles according to a
        # given cubic domain box
        if self.is_periodic and not self.in_parallel:
            # remove periodic ghost particles from a previous step
            self.remove_periodic_ghosts()

            # adjust current particles for periodicity
            self.adjust_particles()

            # create new periodic ghosts
            self.create_periodic_ghosts()

        # indices on which to bin
        for i in range(self.narrays):
            pa = self.particles[i]
            num_particles = pa.get_number_of_particles()
            indices = arange_uint(num_particles)

            # bin the particles
            self._bin( pa_index=i, indices=indices )

    cdef _bin(self, int pa_index, UIntArray indices):
        """Bin a given particle array with indices.

        Parameters:
        -----------

        pa_index : int
            Index of the particle array corresponding to the particles list

        indices : UIntArray
            Subset of particles to bin

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[ pa_index ]
        cdef DoubleArray x = pa_wrapper.x
        cdef DoubleArray y = pa_wrapper.y
        cdef DoubleArray z = pa_wrapper.z

        cdef dict cells = self.cells
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        cdef cIntPoint _cid
        cdef IntPoint cid

        cdef Cell cell
        cdef int ierr, narrays = self.narrays

        # now bin the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            pnt = cPoint_new( x.data[i], y.data[i], z.data[i] )
            _cid = find_cell_id( pnt, cell_size )

            cid = IntPoint_from_cIntPoint(_cid)

            ierr = PyDict_Contains(cells, cid)
            if ierr == 0:
                cell = Cell(cid, cell_size, narrays)
                cells[ cid ] = cell

            # add this particle to the list of indicies
            cell = <Cell>PyDict_GetItem( cells, cid )
            lindices = <UIntArray>PyList_GetItem( cell.lindices, pa_index )

            lindices.append( <ZOLTAN_ID_TYPE> i )
            #gindices.append( gid.data[i] )

    cdef _compute_cell_size(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor. For parallel
        runs, we would need to communicate the maximum 'h' on all
        processors to decide on the appropriate binning size.

        """
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray h
        cdef double cell_size
        cdef double _hmax, hmax = -1.0

        for pa_wrapper in pa_wrappers:
            h = pa_wrapper.h
            h.update_min_max()

            _hmax = h.maximum
            if _hmax > hmax:
                hmax = _hmax

        cell_size = self.radius_scale * hmax

        if cell_size < 1e-6:
            msg = """Cell size too small %g. Perhaps h = 0?
            Setting cell size to 1"""%(cell_size)
            print msg
            cell_size = 1.0

        self.cell_size = cell_size

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        """Utility function to get near-neighbors for a particle.

        Parameters:
        -----------

        src_index : int
            Index of the source particle array in the particles list

        dst_index : int
            Index of the destination particle array in the particles list

        d_idx : int (input)
            Destination particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        cdef dict cells = self.cells
        cdef Cell cell

        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination particle arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size
        cdef UIntArray lindices
        cdef size_t indexj
        cdef ZOLTAN_ID_TYPE j

        cdef cPoint xi = cPoint_new(d_x.data[d_idx], d_y.data[d_idx], d_z.data[d_idx])
        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef IntPoint cid = IntPoint_from_cIntPoint( _cid )
        cdef IntPoint cellid = IntPoint(0, 0, 0)

        cdef cPoint xj
        cdef double xij

        cdef double hi, hj
        hi = radius_scale * d_h.data[d_idx]

        cdef int ierr, nnbrs = 0

        cdef int ix, iy, iz
        for ix in [cid.data.x -1, cid.data.x, cid.data.x + 1]:
            for iy in [cid.data.y - 1, cid.data.y, cid.data.y + 1]:
                for iz in [cid.data.z - 1, cid.data.z, cid.data.z + 1]:
                    cellid.data.x = ix; cellid.data.y = iy; cellid.data.z = iz

                    ierr = PyDict_Contains( cells, cellid )
                    if ierr == 1:

                        cell = <Cell>PyDict_GetItem( cells, cellid )
                        lindices = <UIntArray>PyList_GetItem( cell.lindices, src_index )

                        for indexj in range( lindices.length ):
                            j = lindices.data[indexj]

                            xj = cPoint_new( s_x.data[j], s_y.data[j], s_z.data[j] )
                            xij = cPoint_distance( xi, xj )

                            hj = radius_scale * s_h.data[j]

                            if ( (xij < hi) or (xij < hj) ):
                                if nnbrs == nbrs.length:
                                    nbrs.resize( nbrs.length + 50 )
                                    if self.warn:
                                        print """NNPS:: Extending the neighbor list to %d"""%(nbrs.length)

                                nbrs.data[ nnbrs ] = j
                                nnbrs = nnbrs + 1

        # update the _length for nbrs to indicate the number of neighbors
        nbrs._length = nnbrs


cdef class LinkedListNNPS(NNPS):
    """Nearest neighbor query class using the linked list method.
    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None,
                 bint fixed_h=False, bint warn=True):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Dimension (Not sure if this is really needed)

        particles : list
            The list of particles we are working on

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int
            Optional number of layers to share in parallel

        domain : DomainLimits, default (None)
            Optional limits for the domain
            
        fixed_h : bint
            Optional flag to use constant cell sizes throughout.

        """
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain, warn)

        # initialize the head and next for each particle array
        self.heads = [UIntArray() for i in range(self.narrays)]
        self.nexts = [UIntArray() for i in range(self.narrays)]

        # flag for constant smoothing lengths
        self.fixed_h = fixed_h

        # min and max coordinate values
        self.xmin = DoubleArray(3)
        self.xmax = DoubleArray(3)

        # defaults
        self.ncells = IntArray(3)
        self.ncells_tot = 0

        # cell shifts
        self.cell_shifts = IntArray(3)
        self.cell_shifts.data[0] = -1
        self.cell_shifts.data[1] = 0
        self.cell_shifts.data[2] = 1

        # compute the intial box sort for all local particles
        self.update()

    cpdef update(self):
        """Update the local data after a parallel update.

        We want the NNPS to be independent of the ParallelManager
        which is solely responsible for distributing particles across
        available processors. We assume therefore that after a
        parallel update, each processor has all the local particle
        information it needs.

        """
        cdef int i, num_particles
        cdef ParticleArray pa
        cdef UIntArray indices

        # compute the cell sizes
        self._compute_cell_size()

        # refresh the head and next arrays.
        self._refresh()

        # Periodicity is handled by adjusting particles according to a
        # given cubic domain box
        if self.is_periodic and not self.in_parallel:
            # remove periodic ghost particles from a previous step
            self.remove_periodic_ghosts()

            # box-wrap current particles for periodicity
            self.adjust_particles()

            # create new periodic ghosts
            self.create_periodic_ghosts()

        # indices on which to bin. We bin all local particles
        for i in range(self.narrays):
            pa = self.particles[i]
            num_particles = pa.get_number_of_particles()
            indices = arange_uint(num_particles)

            # bin the particles
            self._bin( pa_index=i, indices=indices )

    cdef _bin(self, int pa_index, UIntArray indices):
        """Bin a given particle array with indices.

        Parameters:
        -----------

        pa_index : int
            Index of the particle array corresponding to the particles list

        indices : UIntArray
            Subset of particles to bin

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[ pa_index ]
        cdef DoubleArray x = pa_wrapper.x
        cdef DoubleArray y = pa_wrapper.y
        cdef DoubleArray z = pa_wrapper.z

        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef IntArray ncells = self.ncells
        cdef int dim = self.dim

        # the head and next arrays for this particle array
        cdef UIntArray head = self.heads[ pa_index ]
        cdef UIntArray next = self.nexts[ pa_index ]
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        # point and flattened index
        cdef cPoint pnt
        cdef int _cid

        # flattened cell index
        cdef int cell_index

        # now bin the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            # the flattened index is considered relative to the
            # minimum along each co-ordinate direction
            pnt = cPoint_new( 
                x.data[i] - xmin.data[0], 
                y.data[i] - xmin.data[1], 
                z.data[i] - xmin.data[2] )

            # flattened cell index
            _cid = flatten( find_cell_id( pnt, cell_size ), ncells, dim )

            # insert this particle
            next.data[ i ] = head.data[ _cid ]
            head.data[_cid] = i                

    cdef _compute_cell_size(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor.

        """
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray h, x, y, z
        cdef double cell_size
        cdef double hmax = -1.0
        cdef double xmax = -1e100, ymax = -1e100, zmax = -1e100
        cdef double xmin = 1e100, ymin = 1e100, zmin = 1e100

        for pa_wrapper in pa_wrappers:
            x = pa_wrapper.x 
            y = pa_wrapper.y
            z = pa_wrapper.z
            h = pa_wrapper.h

            # find min and max of variables
            h.update_min_max()
            x.update_min_max()
            y.update_min_max()
            z.update_min_max()

            hmax = fmax(h.maximum, hmax)
            
            xmax = fmax(x.maximum, xmax)
            ymax = fmax(y.maximum, ymax)
            zmax = fmax(z.maximum, zmax)
            
            xmin = fmin(x.minimum, xmin)
            ymin = fmin(y.minimum, ymin)
            zmin = fmin(z.minimum, zmin)

        # store the minimum and maximum of physical coordinates
        self.xmin.data[0]=xmin; self.xmin.data[1]=ymin; self.xmin.data[2]=zmin
        self.xmax.data[0]=xmax; self.xmax.data[1]=ymax; self.xmax.data[2]=zmax

        # store the cell size
        cell_size = self.radius_scale * hmax
        if cell_size < 1e-6:
            msg = """Cell size too small %g. Perhaps h = 0?
            Setting cell size to 1"""%(cell_size)
            print msg
            cell_size = 1.0

        self.cell_size = cell_size

    cpdef _refresh(self):
        """Refresh the head and next arrays locally"""
        cdef DomainLimits domain = self.domain
        cdef int narrays = self.narrays
        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef double cell_size = self.cell_size
        cdef double cell_size1 = 1./cell_size
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef list heads = self.heads
        cdef list nexts = self.nexts
        cdef int dim = self.dim

        # locals
        cdef int i, j, np
        cdef int ncx, ncy, ncz, _ncells
        cdef UIntArray head, next

        # adjust xmin and xmax in each co-ordinate direction for periodicity
        # if domain.periodic_in_x: xmin.data[0] = xmin.data[0] - 0.1*cell_size
        # if domain.periodic_in_y: xmin.data[1] = xmin.data[1] - 0.1*cell_size
        # if domain.periodic_in_z: xmin.data[2] = xmin.data[2] - 0.1*cell_size

        # if domain.periodic_in_x: xmax.data[0] = xmax.data[0] + 0.1*cell_size
        # if domain.periodic_in_y: xmax.data[1] = xmax.data[1] + 0.1*cell_size
        # if domain.periodic_in_z: xmax.data[2] = xmax.data[2] + 0.1*cell_size

        # calculate the number of cells.
        ncx = <int>ceil( cell_size1*(xmax.data[0] - xmin.data[0]) )
        ncy = <int>ceil( cell_size1*(xmax.data[1] - xmin.data[1]) )
        ncz = <int>ceil( cell_size1*(xmax.data[2] - xmin.data[2]) )

        # number of cells along each coordinate direction
        self.ncells.data[0] = ncx 
        self.ncells.data[1] = ncy 
        self.ncells.data[2] = ncz
        
        # total number of cells
        _ncells = ncx
        if dim == 2: _ncells = ncx * ncy
        if dim == 3: _ncells = ncx * ncy * ncz
        self.ncells_tot = <int>_ncells

        # initialize the head and next arrays
        for i in range(narrays):
            pa_wrapper = pa_wrappers[i]
            np = pa_wrapper.get_number_of_particles()

            # re-size the head and next arrays
            head = <UIntArray>PyList_GetItem(heads, i)
            next = <UIntArray>PyList_GetItem(nexts, i )

            head.resize( _ncells )
            next.resize( np )

            # UINT_MAX is used to indicate an invalid index
            for j in range(_ncells):
                head.data[j] = UINT_MAX

            for j in range(np):
                next.data[j] = UINT_MAX

    cpdef get_cell_indices(
        self, int cell_index, int pa_index, UIntArray indices):
        """Return the indices for this cell"""
        cdef UIntArray head = self.heads[ pa_index ]
        cdef UIntArray next = self.nexts[ pa_index ]
        
        cdef ZOLTAN_ID_TYPE _next
        cdef int num_indices = 0
        
        # get the first particle in this cell
        _next = head.data[ cell_index ]
        while( _next != UINT_MAX ):
            # resize if necessary
            if indices.length == num_indices:
                indices.resize( num_indices + 20 )

            # add to the list of particle indices
            indices.data[num_indices] = _next
            num_indices = num_indices + 1

            _next = next.data[ _next ]
        
        # store the number of indices
        indices._length = num_indices

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        """Utility function to get near-neighbors for a particle.

        Parameters:
        -----------

        src_index : int
            Index of the source particle array in the particles list

        dst_index : int
            Index of the destination particle array in the particles list

        d_idx : int (input)
            Destination particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        # src and dst particle arrays
        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # src and dst linked lists
        cdef UIntArray next = self.nexts[ src_index ]
        cdef UIntArray head = self.heads[ src_index ]

        # Number of cells
        cdef IntArray ncells = self.ncells
        cdef int ncells_tot = self.ncells_tot
        cdef int dim = self.dim

        # cell shifts
        cdef IntArray shifts = self.cell_shifts

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination particle arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        # cell size and radius
        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size

        # locals
        cdef UIntArray lindices
        cdef size_t indexj
        cdef cPoint xj
        cdef double xij
        cdef double hi, hj
        cdef int ierr, nnbrs
        cdef unsigned int _next
        cdef int ix, iy, iz

        # get the un-flattened index for the destination particle
        cdef cPoint xi = cPoint_new(
            d_x.data[d_idx], d_y.data[d_idx], d_z.data[d_idx])
        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef cIntPoint cid = cIntPoint_new(0, 0, 0)
        cdef int cell_index

        # gather search radius
        hi = radius_scale * d_h.data[d_idx]

        # Begin search through neighboring cells
        nnbrs = 0
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    cid.x = _cid.x + shifts.data[ix]
                    cid.y = _cid.y + shifts.data[iy]
                    cid.z = _cid.z + shifts.data[iz]
                    cell_index = flatten( cid, ncells, dim )

                    # only use a valid cell index
                    if -1 < cell_index < ncells_tot:
                      
                        # get the first particle and begin iteration
                        _next = head.data[ cell_index ]
                        while( _next != UINT_MAX ):
                            xj = cPoint_new( 
                                s_x.data[_next], 
                                s_y.data[_next], 
                                s_z.data[_next] 
                                )

                            xij = cPoint_distance( xi, xj )
                            hj = radius_scale * s_h.data[_next]

                            if ( (xij < hi) or (xij < hj) ):
                                if nnbrs == nbrs.length:
                                    nbrs.resize( nbrs.length + 50 )
                                    if self.warn:
                                        print """NNPS:: Extending the neighbor list to %d"""%(nbrs.length)

                                nbrs.data[ nnbrs ] = _next
                                nnbrs = nnbrs + 1

                            # get the 'next' particle in this cell
                            _next = next.data[_next]

        # update the _length for nbrs to indicate the number of neighbors
        nbrs._length = nnbrs

    cpdef get_cell_neighbors(
        self, int cell_index, int pa_index, UIntArray nbrs):
        """Get all potential neighbors from src w.r.t. a cell

        Parameters:
        -----------

        cell_index : int
            Cell index in the range [0,ncells_tot]

        pa_index : int
            Index of the particle array in the particles list

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        # src linked lists
        cdef UIntArray next = self.nexts[ pa_index ]
        cdef UIntArray head = self.heads[ pa_index ]

        # Number of cells in each dimension and the total ncells
        cdef IntArray ncells = self.ncells
        cdef int ncells_tot = self.ncells_tot
        cdef int dim = self.dim

        # cell shifts
        cdef IntArray shifts = self.cell_shifts

        # locals
        cdef size_t indexj
        cdef unsigned int _next
        cdef int nnbrs, ix, iy, iz

        # get the un-flattened index for this cell
        cdef cIntPoint _cid = unflatten( cell_index, ncells, dim )
        cdef cIntPoint cid
        cdef int _cell_index

        tmp = flatten(_cid, ncells, dim)

        # Begin search through neighboring cells
        nnbrs = 0
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    cid.x = _cid.x + shifts.data[ix]
                    cid.y = _cid.y + shifts.data[iy]
                    cid.z = _cid.z + shifts.data[iz]
                    _cell_index = flatten( cid, ncells, dim )
                    
                    # only use a valid cell index
                    if -1 < _cell_index < ncells_tot:
                      
                        # get the first particle and begin iteration
                        _next = head.data[ _cell_index ]
                        while( _next != UINT_MAX ):
                            if nnbrs == nbrs.length:
                                nbrs.resize( nbrs.length + 50 )
                                if self.warn:
                                    print """NNPS:: Extending the potential cell neighbor list to %d"""%(nbrs.length)

                            nbrs.data[ nnbrs ] = _next
                            nnbrs = nnbrs + 1

                            # get the 'next' particle in this cell
                            _next = next.data[_next]

        # update the _length for nbrs to indicate the number of neighbors
        nbrs._length = nnbrs

    cpdef get_nearest_particles_by_cell(
        self,int src_index, int dst_index, int d_idx, UIntArray potential_nbrs,
        UIntArray nbrs):
        """Filter the nearest neighbors from a list of potential neighbors"""

        # src and dst particle arrays
        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]
        
        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination data arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        # cell size and radius_scale 
        cdef double cell_size = self.cell_size
        cdef double radius_scale = self.radius_scale

        # locals
        cdef cPoint xi, xj
        cdef double hi, hj, xij
        cdef int npotential_nbrs = potential_nbrs._length
        cdef int indexj, j
        cdef int nnbrs

        # gather search radius for particle 'i'
        xi = cPoint_new(d_x.data[d_idx], d_y.data[d_idx],  d_z.data[d_idx])
        hi = radius_scale * d_h.data[d_idx]

        # search for true neighbors
        nnbrs = 0
        for indexj in range( npotential_nbrs ):
            j = potential_nbrs.data[indexj]
            
            xj = cPoint_new( s_x.data[j], s_y.data[j], s_z.data[j] )
            hj = radius_scale * s_h.data[j]
            
            xij = cPoint_distance(xi, xj)
       
            if ( (xij < hi) or (xij < hj) ):
                if nnbrs == nbrs.length:
                    nbrs.resize( nbrs.length + 50 )
                    if self.warn:
                        print """NNPS:: Extending the neighbor list to %d"""%(nbrs.length)

                nbrs.data[ nnbrs ] = j
                nnbrs = nnbrs + 1
                
        # store the number of neighbors
        nbrs._length = nnbrs
