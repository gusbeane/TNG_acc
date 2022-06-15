import glob

import arepo
import h5py as h5
from joblib import Parallel, delayed
from numba import njit
import numpy as np
from tqdm import tqdm

def _get_time_integrand(zp, Omega0, OmegaLambda):
    Ezp = np.sqrt(Omega0*(1+zp)**(3.) + OmegaLambda + (1. - Omega0 - OmegaLambda) * (1+zp)**(2.))
    return ((1.+zp)*Ezp)**(-1.)

def get_time(z, Omega0, OmegaLambda):
    # in units of kpc/h/(km/s)
    return 10.*quad(_get_time_integrand, z, np.inf, args=(Omega0, OmegaLambda))[0]

@njit
def find_keys_two_list(t0, t1):
    # Finds the position in t1 that all the items of t0 appear.
    # This requires that all of the elements of t1 are unique - but NOT that all elements of t0 are unique.
    assert len(t1) == len(np.unique(t1)), "Elements of t1 must be unique."
    
    t0_argsort = np.argsort(t0)
    t1_argsort = np.argsort(t1)
    t0_sort = np.sort(t0)
    t1_sort = np.sort(t1)
    N0 = len(t0)
    N1 = len(t1)

    keys = np.full(N0, -1)

    i, j = 0, 0
    while i < N0 and j < N1:
        if t0_sort[i] == t1_sort[j]:
            keys[t0_argsort[i]] = t1_argsort[j]
            i += 1
            continue
        elif t0_sort[i] < t1_sort[j]:
            i += 1
            continue
        elif t0_sort[i] > t1_sort[j]:
            j += 1
            continue
    
    return keys

@njit
def _identify_accreted_excreted(pos0, pos1, Rvir, ID0, ID1, prop0, prop1):
    # this assumes pos0 and pos1 are in _physical coordinates_ relative
    # to the subhalo at snap 0 and snap 1
    
    r0 = np.linalg.norm(pos0)
    r1 = np.linalg.norm(pos1)

    N0 = len(r0)
    N1 = len(r1)
    ID0_sort = np.sort(ID0)
    ID1_sort = np.sort(ID1)
    ID0_argsort = np.argsort(ID0)
    ID1_argsort = np.argsort(ID1)

    accreted_list = np.zeros(np.minimum(N0, N1), dtype=np.int_)
    excreted_list = np.zeros(np.minimum(N0, N1), dtype=np.int_)
    Nacc, Nexc = 0, 0

    i, j = 0, 0
    while i < N0 and j < N1:
        if ID0_sort[i] == ID1_sort[j]:
            if r0[ID0_argsort[i]] >= Rvir and r1[ID1_argsort[j]] < Rvir:
                accreted_list[Nacc] = ID0_sort[i]
                Nacc += 1
            elif r0[ID0_argsort[i]] < Rvir and r1[ID1_argsort[j]] >= Rvir:
                excreted_list[Nexc] = ID0_sort[i]
                Nexc += 1
            i += 1
            j += 1
        elif ID0_sort[i] < ID1_sort[j]:
            i += 1
        else:
            j += 1
    
    accreted_list = accreted_list[:Nacc]
    excreted_list = excreted_list[:Nexc]
    
    return accreted_list, excreted_list

class accretion_calculator(object):
    def __init__(self, basedir, sim, subbox):
        self.basedir = basedir
        self.sim = sim
        self.simdir = basedir + '/' + sim + '/'
        self.subbox = subbox
        if isinstance(self.subbox, int):
            self.subbox = str(self.subbox)

        self._read_files()
    
    def _read_files(self):
        self.fof_length = len(glob.glob(self.simdir + '/output/groups_099/fof_subhalo_tab_099.*.hdf5'))
        self.fof_list = [h5.File(self.simdir+'/output/groups_099/fof_subhalo_tab_099.'+str(i)+'.hdf5', mode='r') \
                         for i in range(self.fof_length)]
    
        # load in circularities and subbox subhalo list
        self.circ = h5.File(self.simdir+'/postprocessing/circularities/circularities_aligned_10Re_'+self.sim.replace('/','')+'099.hdf5', mode='r')
        self.subbox_subhalo = h5.File(self.simdir+'/postprocessing/SubboxSubhaloList/subbox'+self.subbox+'_99.hdf5', mode='r')

    def _read_snap(self, snapnum, fields=['ParentID', 'TracerID', 'Coordinates', 
                               'GFM_Metallicity', 'InternalEnergy', 'Velocities', 'ParticleIDs'],
                       parttype=[0, 3]):
        
        snap_name = self.simdir+'/output/subbox'+self.subbox+'/snapdir_subbox'+self.subbox+'_'+"{:03d}".format(self.snapnum)+'/snap_subbox'+self.subbox+'_'+"{:03d}".format(self.snapnum)+'.0.hdf5'
    
        return arepo.Snapshot(snap_name, combineFiles=True, parttype=parttype, fields=fields)

    def gen_snaps_of_interest(self, zi):
        zi = 1
        ai = 1./(1.+zi)

        ii = 0
        while self.subbox_subhalo['SubboxScaleFac'][ii] < ai:
            ii+=1

        imax = len(self.subbox_subhalo['SubboxScaleFac'])

        snapnum_list = np.arange(ii, imax)

        return snapnum_list
    
    

if __name__ == '__main__':
    # Set some parameters
    basedir = '/n/hernquistfs3/IllustrisTNG/Runs/'
    # sim = '/L35n2160TNG/'
    sim = '/L35n1080TNG/'
    simdir = basedir + sim

    subbox = '0'
    center = np.array([26000, 10000, 26500])
    boxsize = 4000
    hlittle = 0.6774

    nproc = 64

    # Set subhalo idx based on above results
    # TNG50-2
    subhalo_idx = 92418
    Rvir = 229.17804

    # TNG50-1
    # subhalo_idx = 537941
    # Rvir = 229.54263  
