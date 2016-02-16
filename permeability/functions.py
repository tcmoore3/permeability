import numpy as np
import os
#import ipdb
import natsort


def integrate_acf_over_time(filename, average_fraction=0.1):
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
    dataSym_err : np.ndarray, shape=(n,)
        error estimate in symmetrized data

    This function symmetrizes a 1D array. It also provides an error estimate
    for each value, taken as the standard error between the "left" and "right"
    values.
    """
    n_windows = data.shape[0]
    n_win_half = int(np.ceil(float(n_windows)/2))
    dataSym = np.zeros_like(data)
    dataSym_err = np.zeros_like(data)
    for i, sym_val in enumerate(dataSym[:n_win_half]):
        val = 0.5*(data[i] + data[-(i+1)])
        err = np.std([data[i], data[-(i+1)]-data[-1]]) / np.sqrt(2)
        dataSym[i], dataSym_err[i] = val, err
        dataSym[-(i+1)], dataSym_err[-(i+1)] = val, err        
    #dataSym[:] -= dataSym[0]
    return dataSym, dataSym_err

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
    origin = 0 
    for i in range(ntraj):
        dfzt = forces[origin:origin+funlen] - meanfz
        dfz0 = forces[origin] - meanfz;
        f1 += dfzt*dfz0
        origin += dstart
    return f1/ntraj

def analyze_force_acf_data(path, T, n_sweeps=None, verbosity=1, kB=1.9872041e-3,
        directory_prefix='Sweep'):
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
    directory_prefix : str, default = 'Sweep'
        Prefix of directories in path that contain the force ACF data. E.g., if
        the data is in sweep<N>, use directory_prefix='sweep'

    Returns
    -------
    z_windows : np.ndarray, shape=(n_windows,)
        The values of the windows in z
    time : np.ndarray, shape=(n_timepoints,)
        The time values of the correlation functions
    forces : np.ndarray, shape=(n_sweeps, n_windows)
        The forces from each window at each sweep
    dG : np.ndarray, shape(n_sweeps, n_windows)
        The free energy profile from each sweep
    int_facf_win : np.ndarray, shape=(n_windows/2, n_timepoints)
        The values of the integrated autocorrelation functions over time,
        from each window
    dG_mean : np.ndarray, shape=(n_windows,)
        The averaged free energy profiles from all sweeps
    dG_stderr : np.ndarray, shape=(n_windows,)
        The standard error of the free energies at each window
    diffusion_coeff : np.ndarray, shape=(n_windows,)
        The mean diffusion coefficients from each window
    diffusion_coeff_err : np.ndarray, shape=(n_windows,)
        The error estimate on the diffusion coefficients from each window
    dG_sym : np.ndarray, shape=(n_windows,)
        The symmetrized average free energy profile
    dG_sym_err : np.ndarray, shape=(n_windows,)
        The error estimate in the symmetrized average free energy profile
    diff_coeff_sym : np.ndarray, shape=(n_windows,)
        The mean diffusion coefficients from each window
    diff_coeff_sym_err : np.ndarray, shape=(n_windows,)
        The error estimate on the diffusion coefficients from each window
    int_F_acf_vals : np.ndarray
        The integrals of the force autocorrelation functions

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
    RT2 *= 1e5*1e-4  # 10e5 cm2/s
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
        if verbosity >=2:
            print('window / window z-value / max int_F')
        for window in range(n_windows):
            filename = os.path.join(path, sweep_dir, 'fcorr{0}.dat'.format(window))
            int_F, int_F_val = integrate_acf_over_time(filename)
            int_F_acf_vals[sweep, window] = int_F_val
            int_Fs.append(int_F)
            if int_facf_win is None:
                int_facf_win = np.zeros((n_win_half, int_F.shape[0]))
            forces[sweep, window] = np.loadtxt(
                    os.path.join(path, sweep_dir, 'meanforce{0}.dat'.format(window)))
            if verbosity >= 2:
                print(window, z_windows[window], max(int_F))
        for i, val in enumerate(int_facf_win):
            val += 0.5 * (int_Fs[i] + int_Fs[-i-1])
        if verbosity >= 1:
            print('End of sweep {0}'.format(sweep))
        dG[sweep, :] = -4.184 * np.cumsum(forces[sweep,:]) * dz
    int_facf_win /= n_sweeps
    dG_mean = np.mean(dG, axis=0)
    dG_stderr = np.std(dG, axis=0) / np.sqrt(n_sweeps)
    diffusion_coeff = RT2 / np.mean(int_F_acf_vals, axis=0)
    diffusion_coeff_err = np.std(RT2 / int_F_acf_vals, axis=0) / np.sqrt(n_sweeps)
    dG_sym, dG_sym_err = symmetrize(dG_mean) 
    dG_sym -= dG_sym[0] # since the integration (over the forces) starts at 0 
    diff_coeff_sym, diff_coeff_sym_err = symmetrize(diffusion_coeff) 
    #np.savetxt('dGmean.dat', np.vstack((z_windows, dGmeanSym)).T, fmt='%.4f')
    return (z_windows, time, forces, dG, int_facf_win, dG_mean, dG_stderr,
            diffusion_coeff, diffusion_coeff_err, dG_sym, dG_sym_err, diff_coeff_sym, 
            diff_coeff_sym_err, int_F_acf_vals)

def analyze_sweeps(path, n_sweeps=None, correlation_length=300000, 
        verbosity=0, directory_prefix='Sweep'):
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
