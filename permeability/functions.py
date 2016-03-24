import os

import natsort
import numpy as np
import math
from math import factorial
from groupy.gbb import Gbb 
from groupy.system import System
from groupy.mdio import *
from groupy.order import *
from groupy.box import Box

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """ Smooth data
    Parameters
    ----------
    y : np.ndarray, shape=(n,)
        data to be smoothed
    window_size : int
        smoothing width
    order : int
        order of local polynomial fit
    deriv : int
    rate : int

    Returns
    -------
         : np.ndarray, shape=(n,)
    smoothed data, same lengths as original data
    """
    if not (isinstance(window_size, int) and isinstance(order, int)):
        raise ValueError("window_size and order must be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('window_size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('window_size is too small for the polynomials order')

    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window,
                                                           half_window +1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] + np.abs(y[1:half_window+1][::-1] - y[0])  # + should be -, just testing 
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def perm_coeff(z, resist, resist_err):
    """Calculate the overall permeability
    
    Params
    ------
    z : np.ndarray, shape=(n_windows,)
        The position of each window in Angstroms
    resist : np.ndarray, shape=(n_windows,)
        The resistance in each window
    resist_err : np.ndarray, shape=(n_windows,)
        The uncertainty in resistance in each window

    Returns
    -------
    P : int
        Overall permeability of the bilayer
    P_err : int
        Uncertainty in overall permeability of the bilayer
    """
    
    P = 1 / (np.sum(resist) * (z[1] - z[0]) * 1e-8) # convert z from \AA to cm
    P_err = np.sqrt(np.sum(resist_err**2) * (z[1] - z[0]) * 1e-8) * (P**2)
    
    print('Overall permeability: {P:.3e} [cm/s]'.format(**locals()))
    #print('Error in permeability: {P_err:.3e} [cm/s]'.format(**locals()))
    print('WVTR: %f [g/m^2/hr]' % (3.6e7 * P))
    return P, P_err

def integrate_acf_over_time(filename, timestep=1.0, average_fraction=0.1):
    """Open a text file, integrate the forces

    Params
    ------
    filename : str
        Filename of the text file containing forces
    timestep : int
        Simulation timestep in fs
    average_fraction : float
        Use last average_fraction of the data to calculate average 
        value of the converged eagle
    
    Returns
    -------
    intF : np.ndarray, shape=(n,)
        cummulative integral over a function
    intFval : float 
        the average value of the converged integral
    FACF : np.ndarray, shape=(n,)
        force autocorrelation function

    The text file is a 2 column file, with time in the first column and 
    the forces in the second column.
    """
    data =  np.loadtxt(filename)
    time, FACF = data[:,0], data[:,1] # time is stored in ps
    intF = np.cumsum(FACF)*(time[1]-time[0])
    lastbit = int((1.0-average_fraction)*intF.shape[0])
    intFval = np.mean(intF[-lastbit:])
    return intF, intFval, FACF 

def symmetrize_each(data, zero_boundary_condition=False):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,n_sweeps)
        Data to be symmetrized
    zero_boundary_condition : bool, default=False
        If True, shift the right half of the curve before symmetrizing

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values. The zero_boundary_condition shifts the "right" half of the curve 
    such that the final value goes to 0. This should be used if the data is 
    expected to approach zero, e.g., in the case of pulling a water molecule 
    through one phase into bulk water.
    """
    n_sweeps = data.shape[0]
    n_windows = data.shape[1]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    for s in range(n_sweeps):
        for i, sym_val in enumerate(dataSym[s,:n_win_half]):
            val = 0.5 * (data[s,i] + data[s,-(i+1)])
            dataSym[s,i] = val
            dataSym[s,-(i+1)] = val
        if zero_boundary_condition:
            dataSym[s,:] -= dataSym[s,0] 
    return dataSym

def symmetrize(data, zero_boundary_condition=False):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,)
        Data to be symmetrized
    zero_boundary_condition : bool, default=False
        If True, shift the right half of the curve before symmetrizing

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data
    dataSym_err : np.ndarray, shape=(n,)
        error estimate in symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values. The zero_boundary_condition shifts the "right" half of the curve 
    such that the final value goes to 0. This should be used if the data is 
    expected to approach zero, e.g., in the case of pulling a water molecule 
    through one phase into bulk water.
    """
    n_windows = data.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    dataSym_err = np.zeros_like(data)
    shift = {True: data[-1], False: 0.0}
    for i, sym_val in enumerate(dataSym[:n_win_half]):
        val = 0.5 * (data[i] + data[-(i+1)])
        err = np.std([data[i], data[-(i+1)] - shift[zero_boundary_condition]]) / np.sqrt(2)
        dataSym[i], dataSym_err[i] = val, err
        dataSym[-(i+1)], dataSym_err[-(i+1)] = val, err        
    return dataSym, dataSym_err

def acf(forces, timestep, funlen, dstart=10):
    """Calculate the autocorrelation of a function

    Params
    ------
    forces : np.ndarray, shape=(n,)
        The force timeseries acting on a molecules
    timestep : float
        Simulation timestep in fs
    funlen : int
        The desired length of the correlation function

    Returns
    -------
    corr : np.array, shape=(funlen,)
        The autocorrelation of the forces
    """    
    if funlen > forces.shape[0]:
       raise Exception("Not enough data")
    # number of time origins
    ntraj = int(np.floor((forces.shape[0]-funlen)/dstart))
    meanfz = np.mean(forces)
    f1 = np.zeros((funlen))
    origin = 0 
    for i in range(ntraj):
        dfzt = forces[origin:origin+funlen] - meanfz
        dfz0 = forces[origin] - meanfz;
        f1 += dfzt*dfz0
        origin += dstart
    return f1/ntraj

def rotacf(theta, funlen, dstart=2):
    """Calculate the autocorrelation of a function

    Params
    ------
    forces : np.ndarray, shape=(n,)
        The force timeseries acting on a molecules
    timestep : float
        Simulation timestep in fs
    funlen : int
        The desired length of the correlation function

    Returns
    -------
    corr : np.array, shape=(funlen,)
        The autocorrelation of the forces
    """    
    if funlen > theta.shape[0]:
       raise Exception("Not enough data")
    # number of time origins
    ntraj = int(np.floor((theta.shape[0]-funlen)/dstart))
    f1 = np.zeros((funlen))
    origin = 0 
    for i in range(ntraj):
        fzt = theta[origin:origin+funlen]
        fz0 = theta[origin];
        f1 += fzt*fz0
        origin += dstart
    return f1/ntraj

    
def force_timeseries(path, timestep=1.0, n_windows=None, start_window=0, n_sweeps=None, directory_prefix='Sweep'):
    """ Reading the raw data and plotting the sweep-averaged force time series for different windows
    
    Params
    ------
    path : str 
        The path to the directory with the data for each sweep 
    timestep : float
        Simulation timestep in fs
    n_windows : int
        Number of conseqecutive (neighboring) windows to plot, all windows used if no entry
    start_window : int
        The first window to analyze timeseries
    n_sweeps : int
        Number of sweeps to average over, all sweeps are used if no entry
    directory_prefix : str, default = 'Sweep'
        Prefix of directories in path that contain the force ACF data. E.g., if
        the data is in sweep<N>, use directory_prefix='sweep'

    Returns
    -------
    A dict containing the following values:
        time : np.ndarray, shape=(n,)
            The time values of the force timeseries 
        forces : np.ndarray, shape=(n,n_windows)
            sweep-averaged force timeseries in each requested window

    """
    import glob
    sweep_dirs = natsort.natsorted(glob.glob(
        os.path.join(path, '{0}*/'.format(directory_prefix))))
    if n_windows is None:
        z_windows = np.loadtxt(os.path.join(sweep_dirs[0], 'y0list.txt'))
        n_windows = z_windows.shape[0]
    if n_sweeps is None:
        n_sweeps = len(sweep_dirs)
    print(n_windows) 
    serieslen = 150000 # used for force timeseries plot, does not affect permeability calc. 
    forceseries = np.zeros((serieslen, n_windows))
    for sweep, sweep_dir in enumerate(sweep_dirs[:n_sweeps]): 
        for iw, window in enumerate(range(start_window,n_windows+start_window)):
            print('Main Loop:  sweep '+str(sweep)+', window '+str(window))
            data = np.loadtxt(os.path.join(sweep_dir, 'forceout{0}'.format(window)))
            # note that the timeseries are not always of the same length
            # we are primarily interested in the beginning, 
            # to validate equilibrium
            # for now I hardcode a fixed length
            forces = data[range(serieslen), 1]
            dstep = data[1, 0] - data[0, 0]
            
            forceseries[:,iw] += forces/n_sweeps 
    time = data[range(serieslen), 0]*timestep/1000
    
    #np.savetxt('dGmean.dat', np.vstack((z_windows, dGmeanSym)).T, fmt='%.4f')
    return {'time': time, 'forces': forceseries}


def analyze_force_acf_data(path, T, timestep=1.0, n_sweeps=None, verbosity=1, kB=1.9872041e-3,
        directory_prefix='Sweep'):
    """Combine force autocorrelations to calculate the free energy profile
    
    Params
    ------
    path : str
        The path to the directories holding the correlation functions
    T : float
        The absolute temperature
    timestep : float
        Simulation timestep in fs
    kB : float
        Boltzmann constant, determines units (default is kcal/mol-K)
    n_sweeps : int
        The number of sweeps to analyze
    verbosity : int
        Give more details with higher number. 0 - don't print any status
        1 - Print when finished with a sweep
        2 - Print each window of each sweep
    directory_prefix : str, default = 'Sweep'
        Prefix of directories in path that contain the force ACF data. E.g., if
        the data is in sweep<N>, use directory_prefix='sweep'

    Returns
    -------
    A dict containing the following values: 
        z : np.ndarray, shape=(n_windows,)
            The values of the windows in z
        time : np.ndarray, shape=(n_timepoints,)
            The time values of the correlation functions
        forces : np.ndarray, shape=(n_sweeps, n_windows)
            The forces from each window at each sweep
        dG : np.ndarray, shape(n_sweeps, n_windows)
            The free energy profile from each sweep
        int_facf_windows : np.ndarray, shape=(n_windows/2, n_timepoints)
            The values of the integrated autocorrelation functions over time,
            from each window
        facf_windows : np.ndarray, shape=(n_windows/2, n_timepoints)
            The autocorrelation functions over time,
            from each window
        dG_mean : np.ndarray, shape=(n_windows,)
            The averaged free energy profiles from all sweeps
        dG_stderr : np.ndarray, shape=(n_windows,)
            The standard error of the free energies at each window
        d_z : np.ndarray, shape=(n_windows,)
            The mean diffusion coefficients from each window
        d_z_err : np.ndarray, shape=(n_windows,)
            The error estimate on the diffusion coefficients from each window
        dG_sym : np.ndarray, shape=(n_windows,)
            The symmetrized average free energy profile
        dG_sym_err : np.ndarray, shape=(n_windows,)
            The error estimate in the symmetrized average free energy profile
        d_z_sym : np.ndarray, shape=(n_windows,)
            The mean diffusion coefficients from each window
        d_z_sym_err : np.ndarray, shape=(n_windows,)
            The error estimate on the diffusion coefficients from each window
        R_z : np.ndarray, shape=(n_windows,)
            Resistance in each window
        R_z_err : np.ndarray, shape=(n_windows,)
            Uncertainty of resistance in each window
        int_F_acf_vals : np.ndarray
            The integrals of the force autocorrelation functions
        permeability : float
            The global bilayer permeability
        perm_err : float
            The uncertainty in the permeability calculation
    This works under the assumption that the data for each sweep is listed in
    path/SweepN, where N is the sweep number.
    If n_.sweeps=None, then this function finds the number of sweeps.
    Othwerswise, you can pass a number of sweeps to anaylyze, which can be 
    useful for quick testing.
    """
    import glob
    sweep_dirs = natsort.natsorted(glob.glob(
        os.path.join(path, '{0}*/'.format(directory_prefix))))
    time = np.loadtxt(os.path.join(sweep_dirs[0], 'fcorr0.dat'))[:, 0]
    z_windows = np.loadtxt(os.path.join(sweep_dirs[0], 'y0list.txt'))
    n_windows = z_windows.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dz = z_windows[2]-z_windows[1]
    RT2 = (kB*T)**2
    RT2 *= 1e-4  # convert from \AA2/ps to cm2/s
    # arrays to hold the forces, integrated FACFs, and free energies from each
    # window and each sweep
    if n_sweeps is None:
        n_sweeps = len(sweep_dirs)
    forces = np.zeros((n_sweeps, n_windows))
    int_F_acf_vals = np.zeros((n_sweeps, n_windows))
    dG = np.zeros((n_sweeps, n_windows))
    int_facf_win = None
    for sweep, sweep_dir in enumerate(sweep_dirs[:n_sweeps]): 
        int_Fs = []
        Facfs = []
        if verbosity >=2:
            print('window / window z-value / max int_F')
        for window in range(n_windows):
            filename = os.path.join(sweep_dir, 'fcorr{0}.dat'.format(window))
            int_F, int_F_val, Facf = integrate_acf_over_time(filename,timestep)
            int_F_acf_vals[sweep, window] = int_F_val
            int_Fs.append(int_F)
            Facfs.append(Facf)
            if int_facf_win is None:
                int_facf_win = np.zeros((n_win_half, int_F.shape[0]))
                facf_win = np.zeros((n_win_half, int_F.shape[0]))
            forces[sweep, window] = np.loadtxt(
                    os.path.join(sweep_dir, 'meanforce{0}.dat'.format(window)))
            if verbosity >= 2:
                print(window, z_windows[window], max(int_F))
        for i, val in enumerate(int_facf_win):
            val += 0.5 * (int_Fs[i] + int_Fs[-i-1])
            facf_win[i] += 0.5 * (Facfs[i] + Facfs[-i-1])
        if verbosity >= 1:
            print('End of sweep {0}'.format(sweep))
        dG[sweep, :] = - np.cumsum(forces[sweep,:]) * dz
    
    int_facf_win /= n_sweeps
    facf_win /= n_sweeps
    dG_mean = np.mean(dG, axis=0)
    dG_stderr = np.std(dG, axis=0) / np.sqrt(n_sweeps)
    
    diffusion_coeff = RT2 / np.mean(int_F_acf_vals, axis=0)
    diffusion_coeff_err = np.std(RT2 * int_F_acf_vals, axis=0) / np.sqrt(n_sweeps) # incorrect!
   
    int_facf_sym_all = symmetrize_each(int_F_acf_vals) 
    diff_coeff_sym = RT2/np.mean(int_facf_sym_all, axis=0)
    diff_coeff_sym_err = RT2*np.std(int_facf_sym_all, axis=0) / (np.mean(int_facf_sym_all, axis=0)**2) / np.sqrt(n_sweeps)
     
    dG_sym_all = symmetrize_each(dG, zero_boundary_condition=True) 
    dG_sym = np.mean(dG_sym_all, axis=0)
    dG_sym_err = np.std(dG_sym_all, axis=0) / np.sqrt(n_sweeps)
    
    resist_all = np.exp(dG_sym_all / (kB*T)) * int_facf_sym_all / RT2 
    
    expdGerr = np.exp(dG_sym / (kB*T)) * dG_sym_err / (kB*T) 
    resist = np.exp(dG_sym / (kB*T)) / diff_coeff_sym
    resist_err = resist * np.sqrt((expdGerr/np.exp(dG_sym / (kB*T)))**2+(diff_coeff_sym_err/diff_coeff_sym)**2) 
    
    perm, perm_err = perm_coeff(z_windows, resist, resist_err)

    #np.savetxt('dGmean.dat', np.vstack((z_windows, dGmeanSym)).T, fmt='%.4f')
    return {'z': z_windows, 'time': time, 'forces': forces, 'dG': dG,
            'int_facf_windows': int_facf_win, 'facf_windows': facf_win,
            'dG_mean': dG_mean, 
            'dG_stderr': dG_stderr, 'd_z': diffusion_coeff, 
            'd_z_err': diffusion_coeff_err, 'dG_sym': dG_sym, 
            'dG_sym_err': dG_sym_err, 'd_z_sym': diff_coeff_sym,
            'd_z_sym_err': diff_coeff_sym_err, 
            'R_z_all': resist_all, 'R_z': resist, 'R_z_err': resist_err,
            'int_F_acf_vals': int_F_acf_vals, 
            'permeability': perm,'perm_err': perm_err}


def analyze_rotacf_data(path, n_sweeps=None, verbosity=1, directory_prefix='Sweep'):

    import glob
    sweep_dirs = natsort.natsorted(glob.glob(
        os.path.join(path, '{0}*/'.format(directory_prefix))))
    time = np.loadtxt(os.path.join(sweep_dirs[0], 'rcorr0.dat'))[:, 0]
    z_windows = np.loadtxt(os.path.join(sweep_dirs[0], 'y0list.txt'))
    n_windows = z_windows.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))

    if n_sweeps is None:
        n_sweeps = len(sweep_dirs)
    racf_win = None
    for sweep, sweep_dir in enumerate(sweep_dirs[:n_sweeps]): 
        Racfs = []
        if verbosity >=2:
            print('window / window z-value')
        for window in range(n_windows):
            filename = os.path.join(sweep_dir, 'rcorr{0}.dat'.format(window))
            data =  np.loadtxt(filename)
            time, Racf = data[:,0], data[:,1] # time is stored in ps
            Racfs.append(Racf)
            if racf_win is None:
                racf_win = np.zeros((n_win_half, Racf.shape[0]))
            if verbosity >= 2:
                print(window, z_windows[window])
        for i, val in enumerate(racf_win):
            val += 0.5 * (Racfs[i] + Racfs[-i-1])
        if verbosity >= 1:
            print('End of sweep {0}'.format(sweep))
    racf_win /= n_sweeps

    return {'z': z_windows, 'time': time, 'racf_windows': racf_win}


def analyze_sweeps(path, n_sweeps=None, timestep=1.0, correlation_length=300, 
        verbosity=0, directory_prefix='Sweep'):
    """Analyze the force data to calculate the force ACFs and mean force 
    at each window for each sweep

    Params
    ------
    path : str 
        The path to the directory with the data for each sweep 
    n_sweeps : int
        The number of sweeps to analyze
    timestep : float
        Simulation timestep in fs
    correlation_length : float
        Desired force autocorrelation length in ps
    verbosity : int
        Level of detail to print
    directory_prefix : str, default = 'Sweep'
        Prefix of directories in path that contain the force ACF data. E.g., if
        the data is in sweep<N>, use directory_prefix='sweep'

    Returns
    -------
    This function prints the meanforce and force ACF at each window from each 
    sweep.
    """
    import glob
    sweep_dirs = natsort.natsorted(glob.glob(os.path.join(
        path, '{0}*/'.format(directory_prefix))))
    n_windows = np.loadtxt(os.path.join(sweep_dirs[0], 'y0list.txt')).shape[0]
    # loop over sweeps
    for sweep_dir in sweep_dirs[:n_sweeps]:
        if verbosity >= 2:
            print('Window / Mean force / n_timepoints / dstep')
        for window in range(n_windows):
            data = np.loadtxt(os.path.join(sweep_dir, 'forceout{0}'.format(window)))
            forces = data[:, 1]
            dstep = (data[1, 0] - data[0, 0])*timestep/1000 # data intervals in ps 
            if verbosity >= 2:
                print('{0} / {1} / {2} / {3}'.format(
                    window, np.mean(data[:, 1]), data.shape[0], dstep))
            funlen = int(correlation_length/dstep)
            FACF = acf(data[:, 1], timestep, funlen)
            time = np.arange(0, funlen*dstep, dstep) 
            np.savetxt(os.path.join(sweep_dir, 'fcorr{0}.dat'.format(window)),
                    np.vstack((time, FACF)).T, fmt='%.3f')
            np.savetxt(os.path.join(sweep_dir, 'meanforce{0}.dat'.format(window)),
                    [np.mean(data[:, 1])], fmt='%.4f')
        if verbosity >= 1:
            print('Finished analyzing data in {0}'.format(sweep_dir))


def analyze_rot_sweeps(path, n_sweeps=None, correlation_length=300, directory_prefix='Sweep'):
    """Analyze the rotational correlation at each window for each sweep
    Data currently only available for DSPC

    Params
    ------
    path : str 
        The path to the directory with the data for each sweep 
    n_sweeps : int
        The number of sweeps to analyze
    correlation_length : float
        Desired force autocorrelation length in ps
    directory_prefix : str, default = 'Sweep'
        Prefix of directories in path that contain the force ACF data. E.g., if
        the data is in sweep<N>, use directory_prefix='sweep'

    Returns
    -------
    This function prints the rotational ACF at each window from each sweep.
    """
    import glob
    sweep_dirs = natsort.natsorted(glob.glob(os.path.join(
        path, '{0}*/'.format(directory_prefix))))

    if n_sweeps is None:
        n_sweeps = len(sweep_dirs)

    n_windows = np.loadtxt(os.path.join(sweep_dirs[0], 'y0list.txt')).shape[0]
    info = [(7, 3, 'water')] # 7 water molecules per simulation 
    
    # loop over sweeps
    for sweep_dir in sweep_dirs[:n_sweeps]:
        window_order = np.arange(0, 31, 5)
        theta_by_window = list() 
        for i in range(n_windows):
            theta_by_window.append(list())

        for sim in range(5):
            with open(sweep_dir+'tracerpos'+str(sim+1)+'.xyz', 'r') as trj:
                IDs = read_frame_lammpstrj(trj, getIDs=True)
                IDs = np.sort(IDs)
                IDdic = []
                for iID, ID in enumerate(IDs):
                    IDdic.append((ID, iID))
                d_ID = {int(x[0]): x[1] for x in IDdic}
                while True:
                    try:
                        xyz, types, step, box = read_frame_lammpstrj(trj, IDdic=d_ID)
                    except:
                        print('Reached end of file')
                        break
                    system = System(system_info=info, box=box)
                    system.convert_from_traj(xyz, types)
                    for i, gbb in enumerate(system.gbbs):
                        gbb.masses = np.asarray([15.994, 1.008, 1.008]).reshape((3, 1))
                        gbb.calc_com()
                    sorted_gbbs = sorted(system.gbbs, key=lambda x: x.com[2])
                    
                    for window, gbb in zip(window_order, sorted_gbbs):
                        gbb.load_xml_prototype('water.xml', skip_coords=True, skip_types=False, skip_masses=True)
                        director = calc_director(Gbb.calc_inertia_tensor(gbb))
                        
                        theta_by_window[window].append(math.degrees(math.acos(director[2]))-90)
                        #theta_by_window[window].append(director[2])
            window_order += 1

        # all data for this sweep has been collected and can be processed
        print('Window')
        for window in range(n_windows):
            
            dstep = 1.0 # ps
            
            #print('{0}'.format(window))
            funlen = int(correlation_length/dstep)
            
            RACF = rotacf(np.asarray(theta_by_window[window]), funlen)
            time = np.arange(0, funlen*dstep, dstep) 
            np.savetxt(os.path.join(sweep_dir, 'rcorr{0}.dat'.format(window)),
                    np.vstack((time, RACF)).T, fmt='%.3f')
            np.savetxt(os.path.join(sweep_dir, 'orient{0}.dat'.format(window)),
                    np.vstack(theta_by_window[window]).T, fmt='%.3f')
            print(window, np.mean(theta_by_window[window]),np.std(theta_by_window[window]))
        print('Finished analyzing data in {0}'.format(sweep_dir))
