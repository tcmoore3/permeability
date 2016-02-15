import numpy as np
import os
#import ipdb
import natsort


def integrate_forces_over_time(filename, average_fraction=0.1):
    """Open a text file, integrate the forces

    Params
    ------
    filename : str
        Filename of the text file containing forces
    average_fraction : float
        Use last average_fraction of the data to calculate average 
        value of the converged eagle
    
    Returns
    -------
    intF : np.ndarray, shape=(n,)
        cummulative integral over a function
    intFval : float 
        the average value of the converged integral

    The text file is a 2 column file, with time in the first column and 
    the forces in the second column.
    """
    data =  np.loadtxt(filename)
    time, FACF = data[:,0], data[:,1]
    intF = np.cumsum(FACF)*(time[1]-time[0])
    lastbit = int((1.0-average_fraction)*intF.shape[0])
    intFval = np.mean(intF[-lastbit:])
    return intF, intFval 

def symmetrize(data):
    """Symmetrize a profile
    
    Params
    ------
    data : np.ndarray, shape=(n,)
        data to be symmetrized

    Returns
    -------
    dataSym : np.ndarray, shape=(n,)
        symmetrized data
    """
    dataSym = np.zeros_like(data)
    for i, point in enumerate(data):
        dataSym[i] = (point + data[-(i+1)])/2        
    dataSym[:] -= dataSym[0]
    return dataSym

def acf(forces, funlen, dstart=10):
    """Calculate the autocorrelation of a function

    Params
    ------
    forces : np.ndarray, shape=(n,)
        The force timeseries acting on a molecules
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
    f2 = np.zeros((funlen))
    origin = 0 
    for i in range(ntraj):
        dfzt = forces[origin:origin+funlen] - meanfz
        dfz0 = forces[origin] - meanfz;
        f1 += dfzt*dfz0
        f2 += dfz0*dfz0
        origin += dstart
    return f1/ntraj

def combine(path, T, n_sweeps=None, verbosity=1, kB=1.9872041e-3):
    """Combine force autocorrelations to calculate the free energy profile
    
    Params
    ------
    path : str
        The path to the directories holding the correlation functions
    T : float
        The absolute temperature
    kB : float
        Boltzmann constant, determines units (default is kcal/mol-K)
    n_sweeps : int
        The number of sweeps to analyze
    verbosity : int
        Give more details with higher number. 0 - don't print any status
        1 - Print when finished with a sweep
        2 - Print each window of each sweep

    Returns
    -------
    forces : np.ndarray
        The forces from each window at each sweep
    ifacfval : np.ndarray
        The values of the integratd force autocorrelation functions
    delG : np.ndarray
        The calculated free energy profiles for each window at each sweep

    This works under the assumption that the data for each sweep is listed in
    path/SweepN, where N is the sweep number.
    If n_sweeps=None, then this function finds the number of sweeps.
    Othwerswise, you can pass a number of sweeps to anaylyze, which can be 
    useful for quick testing.
    """
    RT2 = (kB*T)**2  # kcal/mol
    RT2 *= 1e5*1e-4  # 10e5 cm2/s
    # get values of the z-windows, calculate window spacing
    z_windows = np.loadtxt(os.path.join(path, 'Sweep0', 'y0list.txt'))
    n_windows = z_windows.shape[0]
    dz = z_windows[2]-z_windows[1]
    n_win_half = int(np.ceil(float(n_windows)/2))
    if n_sweeps is None:
        import glob
        sweep_dirs = natsort.natsorted(glob.glob(os.path.join(path, 'Sweep*')))
        n_sweeps = len(sweep_dirs)
    # use first sweep to load the time
    time = np.loadtxt(os.path.join(sweep_dirs[0], 'fcorr0.dat'))[:, 0]

    # arrays to hold the forces, integrated FACFs, and free energies from each
    # window and each sweep
    forces = np.zeros((n_sweeps,n_windows))
    ifacfval = np.zeros((n_sweeps,n_windows))
    delG = np.zeros((n_sweeps,n_windows))
    int_facf_win = None
    for sweep in range(n_sweeps): 
        int_Fs = []
        for window in range(n_windows):
            filename = os.path.join(
                    path, 'Sweep{0}'.format(sweep), 'fcorr{0}.dat'.format(window))
            int_F, int_F_val = integrate_forces_over_time(filename)
            ifacfval[sweep, window] = int_F_val
            int_Fs.append(int_F)
            if int_facf_win is None: # make an int_facf_win for each sweep
                int_facf_win = np.zeros((n_win_half, int_F.shape[0]))
            forces[sweep, window] = np.loadtxt(
                    os.path.join(path, 'Sweep{0}'.format(sweep), 
                        'meanforce{0}.dat'.format(window)))
            if verbosity >= 2:
                print(window, z_windows[window], max(int_F))
        for i, val in enumerate(int_facf_win):
            val += 0.5 * (int_Fs[i] + int_Fs[-i-1])
        if verbosity >= 1:
            print('End of sweep {0}'.format(sweep))
        delG[sweep, :] = -4.184 * np.cumsum(forces[sweep,:]) * dz
    int_facf_win /= n_sweeps
    dG_mean = np.mean(delG, axis=0)
    diffusion_coefficient = RT2 / np.mean(ifacfval, axis=0)
    dGmeanSym = symmetrize(dG_mean) 
    np.savetxt('dGmean.dat', np.vstack((z_windows, dGmeanSym)).T, fmt='%.4f')
    return (z_windows, time, forces, delG, int_facf_win, dG_mean, 
            diffusion_coefficient, dGmeanSym)

def analyze_sweeps(path, n_sweeps=None, correlation_length=300000, 
        verbosity=0):
    """Analyze the force data to calculate the force ACFs and mean force 
    at each window for each sweep

    Params
    ------
    path : str 
        The path to the directory with the data for each sweep 
    n_sweeps : int
        The number of sweeps to analyze
    verbosity : int
        Level of detail to print

    Returns
    -------
    This function prints the meanforce and force ACF at each window from each sweep.
    """
    import glob
    sweep_dirs = natsort.natsorted(glob.glob(os.path.join(path, 'Sweep*')))
    n_windows = np.loadtxt(os.path.join(sweep_dirs[0], 'y0list.txt')).shape[0]
    # loop over sweeps
    for sweep_dir in sweep_dirs[:n_sweeps]:
        if verbosity >= 2:
            print('Window / Mean force / n_timepoints / dstep')
        for window in range(n_windows):
            data = np.loadtxt(os.path.join(sweep_dir, 'forceout{0}'.format(window)))
            forces = data[:, 1]
            dstep = data[1, 0] - data[0, 0]  # 1 fs per step
            if verbosity >= 2:
                print('{0} / {1} / {2} / {3}'.format(
                    window, np.mean(data[:, 1]), data.shape[0], dstep))
            funlen = int(correlation_length/dstep)
            FACF = acf(data[:, 1], funlen)
            time = np.arange(0, funlen*dstep/1000, dstep/1000)  # WHY 1000?
            np.savetxt(os.path.join(sweep_dir, 'fcorr{0}.dat'.format(window)),
                    np.vstack((time, FACF)).T, fmt='%.4f')
            np.savetxt(os.path.join(sweep_dir, 'meanforce{0}.dat'.format(window)),
                    [np.mean(data[:, 1])], fmt='%.4f')
        if verbosity >= 1:
            print('Finished analyzing data in {0}'.format(sweep_dir))
