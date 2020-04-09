'''Read in DXchange file and reset chunking by image row.
'''
import h5py

def rechunk_dxfile(old_fname, new_fname):
    '''Reads in HDF5 file and writes a copy with the chunking set
        so one chunk encompasses all data for one row of the 
        tomography dataset.
    Inputs:
    old_fname: name of the DXchange file to be rechunked
    new_fname: name of the new DXchange file to be created.
    '''
    with h5py.File(old_fname, 'r') as old_hdf:
        with h5py.File(new_fname, 'w') as new_hdf:
            old_hdf.copy(old_hdf['/defaults'], new_hdf)
            old_hdf.copy(old_hdf['/measurements'], new_hdf)
            old_hdf.copy(old_hdf['/process'], new_hdf)
