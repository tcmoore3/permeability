import matplotlib.pyplot as plt
from permeability.functions import savitzky_golay 
import numpy as np
import pdb

def plot_forces(z_windows, forces, fig_filename='forces.pdf',
        z_units=u'\u00c5', force_units=u'kcal/mol-\u00c5', plot_mean=True,
        sweep_alpha=0.5, grid=True):
    """Plot the forces from analyzing the force data

    Params
    ------
    z_windows : np.ndarary, shape=(n,)
        The z_values for the forces
    forces : np.ndarray, shape=(m, n)
        The forces at each window from each sweep
    fig_filename : str
        Name of file to save, default=forces.pdf'
    z_units : str
        Units of z, default=u'\u00c5'
    force_units : str
        Force units, default=u'kcal/mol-\u00c5'
    plot_mean : bool
        Plot the mean value of the forces if True, default=True
    sweep_alpha : float
        Opacity of line for individual sweeps, default=0.3
    grid : bool
        Plot gridlines on major ticks if True, default=True

    Returns
    -------
    This function saves a figure of the forces from each sweep as a function
    of z position.
    """
    if forces.ndim == 1:  # only has data from 1 sweep
        forces = forces.reshape((1, -1))
    fig, ax = plt.subplots()
    if plot_mean:
        mean_force = np.mean(forces, axis=0)
        ax.plot(z_windows, mean_force)
        std_err = np.std(forces, axis=0) / np.sqrt(forces.shape[0])
        ax.fill_between(z_windows, mean_force+std_err, mean_force-std_err,
                facecolor='#a8a8a8', edgecolor='#a8a8a8')
    for force_series in forces:
        ax.plot(z_windows, force_series, alpha=sweep_alpha, zorder=0)
    ax.set_xlabel(u'z [{z_units}]'.format(**locals()))
    ax.set_ylabel(u'F(z) [{force_units}]'.format(**locals()))
    ax.grid(grid)
    zmin = z_windows[0]    
    plt.xlim(zmin,-zmin)
    fig.tight_layout()
    fig.savefig('{fig_filename}'.format(**locals()))
    plt.show()

def plot_free_energy_z(z_windows, free_energy, fig_filename='delta_G.pdf',
        z_units=u'\u00c5', energy_units=u'kcal/mol', plot_mean=True,
        sweep_alpha=0.5, grid=True):
    """Plot the free energy profile

    Params
    ------
    z_windows : np.ndarary, shape=(n,)
        The z_values for the free energy profile
    free_energy : np.ndarray, shape=(m, n)
        The free_energy at each window from each sweep
    fig_filename : str
        Name of file to save, default=delta_G.pdf'
    z_units : str
        Units of z, default=u'\u00c5'
    energy_units : str
        Energy units, default=u'kcal/mol'
    plot_mean : bool
        Plot the mean value of the free energy if True, default=True
    sweep_alpha : float
        Opacity of line for individual sweeps, default=0.3
    grid : bool
        Plot gridlines on major ticks if True, default=True

    Returns
    -------
    This function saves a figure of the free energy from each sweep as a
    function of z position.
    """
    if free_energy.ndim == 1:  # only has data from 1 sweep
        free_energy = free_energy.reshape((1, -1))
    fig, ax = plt.subplots()
    if plot_mean:
        mean_free_energy = np.mean(free_energy, axis=0)
        ax.plot(z_windows, mean_free_energy)
        std_err = np.std(free_energy, axis=0) / np.sqrt(free_energy.shape[0])
        ax.fill_between(z_windows, mean_free_energy+std_err, 
                mean_free_energy-std_err,
                facecolor='#a8a8a8', edgecolor='#a8a8a8')
    for free_energy_series in free_energy:
        ax.plot(z_windows, free_energy_series, alpha=sweep_alpha, zorder=0)
    ax.set_xlabel(u'z [{z_units}]'.format(**locals()))
    ax.set_ylabel(u'\u0394G(z) [{energy_units}]'.format(**locals()))
    ax.grid(grid)
    zmin = z_windows[0]    
    plt.xlim(zmin,-zmin)
    fig.tight_layout()
    fig.savefig(fig_filename)
    plt.show()


def plot_timeseries(time, forces, time_units='ps', force_units=u'kcal/mol-\u00c5', 
        grid=True, fig_filename='force_timeseries.png'):
    fig, ax = plt.subplots()
    for force_series in forces.T:
        ax.plot(time, force_series,zorder=0)
        smoothdata = savitzky_golay(force_series, 15001, 3)
        ax.plot(time, smoothdata)
    ax.set_xlabel(u'time [{time_units}]'.format(**locals()))
    ax.set_ylabel(u'F_z(z) [{force_units}]'.format(**locals()))
    ax.grid(grid,color='c')
    fig.tight_layout()
    fig.savefig('{fig_filename}'.format(**locals()))
    plt.show()
    
def plot_force_acfs_time(time, facfs, time_units='ps', normalize=True, grid=True,
        fig_filename='acf_per_window.pdf'):
    fig, ax = plt.subplots()
    for facf in facfs:
        if normalize:
            facf /= facf[0]
        ax.semilogx(time, facf)      
    ax.set_xlabel('t [{0}]'.format(time_units))
    ax.set_ylabel(r'$\langle\Delta$F(t)$\Delta$F(0)$\rangle$')
    plt.xlim(time[0],time[-1])
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_int_acfs_time(time, int_facfs, time_units='ps', grid=True,
        fig_filename='int_acf_per_window.pdf'):
    fig, ax = plt.subplots()
    for int_facf in int_facfs:
        ax.loglog(time, int_facf)      
    ax.set_xlabel('t [{0}]'.format(time_units))
    ax.set_ylabel(r"$\int_0^t\langle\Delta$F(t')$\Delta$F(0)$\rangle$dt'")
    plt.xlim(time[0],time[-1])
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_resistance_z(z_windows, resist, 
        z_units=u'\u00c5', Res_units=u's/cm\u00b2', fig_filename='res_z.pdf',
        grid=True):
    """Plot the diffusion coefficient as a function of z-position.
        Resistant input is in 1e-5 s/cm2

    """
    fig, ax = plt.subplots()
    ax.plot(z_windows, resist)
    #ax.fill_between(z_windows, resist+resist_err, 
    #        resist-resist_err,
    #        facecolor='#a8a8a8', edgecolor='#a8a8a8')
    ax.set_xlabel(u'z [{0}]'.format(z_units))
    ax.set_ylabel(u'R(z) [{0}]'.format(Res_units))
    ax.grid(grid)
    zmin = z_windows[0]    
    plt.xlim(zmin,-zmin)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_diffusion_coefficient_z(z_windows, diffusion_coeff, diffusion_coeff_err, 
        z_units=u'\u00c5', D_units=u'cm\u00b2/s', fig_filename='d_z.pdf',
        grid=True):
    """Plot the diffusion coefficient as a function of z-position.
    """
    zmin = z_windows[0]    
    fig, ax = plt.subplots()
    ax.plot(z_windows, diffusion_coeff)
    ax.plot([zmin, zmin+10],[3.86e-5, 3.86e-5],linestyle='--', color='r')
    ax.plot([-zmin-10, -zmin],[3.86e-5, 3.86e-5],linestyle='--', color='r')
    ax.fill_between(z_windows, diffusion_coeff+diffusion_coeff_err, 
            diffusion_coeff-diffusion_coeff_err,
            facecolor='#a8a8a8', edgecolor='#a8a8a8')
    ax.set_xlabel(u'z [{0}]'.format(z_units))
    ax.set_ylabel(u'D(z) [{0}]'.format(D_units))
    plt.ylim(0,3e-4)
    plt.xlim(zmin,-zmin)
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_sym_diffusion_coefficient_z(z_windows, diffusion_coeff, diffusion_coeff_err, 
        z_units=u'\u00c5', D_units=u'cm\u00b2/s', fig_filename='d-sym_z.pdf',
        grid=True):
    """Plot the diffusion coefficient as a function of z-position.
    """
    fig, ax = plt.subplots()
    ax.semilogy(z_windows, diffusion_coeff)
    # from Raabe and Sadus, JCP, 2012
    zmin = z_windows[0]    
    ax.plot([zmin, zmin+10],[3.86e-5, 3.86e-5],linestyle='--', color='r')
    ax.plot([-zmin-10, -zmin],[3.86e-5, 3.86e-5],linestyle='--', color='r')
    ax.fill_between(z_windows, diffusion_coeff+diffusion_coeff_err, 
            diffusion_coeff-diffusion_coeff_err,
            facecolor='#a8a8a8', edgecolor='#a8a8a8')
    ax.set_xlabel(u'z [{0}]'.format(z_units))
    ax.set_ylabel(u'D(z) [{0}]'.format(D_units))
    plt.xlim(zmin,-zmin)
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_symmetrized_free_energy(z_windows, delta_G, delta_G_err, z_units=u'\u00c5',
        energy_units=u'kcal/mol', fig_filename='delG-sym.pdf', grid=True):
    """Plot symmetrized delta G
    
    Params
    ------
    z_windows : np.ndarray, shape=(n,)
        The location of the windows in z
    delta_G : np.ndarray, shape=(n,)
        The symmetrized free energy profile, in energy units
    delta_G_err : np.ndarray, shape=(n,)
        The error in the symmetrized free energy profile, in energy units
    z_units : str
        The units of the z-values in z_windows
    energy_units : str
        The units of delta_G
    fig_filename : str
        The name of the figure file to write
    grid : bool
        Draw gridlines on major ticks if True

    Returns
    -------
    None. This figure draws a figure of the symmetrized free energy profile
    and saves it to disk.
    """

    fig, ax = plt.subplots()
    ax.plot(z_windows, delta_G)
    ax.fill_between(z_windows, delta_G+delta_G_err, 
            delta_G-delta_G_err,
            facecolor='#a8a8a8', edgecolor='#a8a8a8')
    ax.set_xlabel(u'z [{0}]'.format(z_units))
    ax.set_ylabel(u'\u0394G(z) [{0}]'.format(energy_units))
    ax.grid(grid)
    zmin = z_windows[0]    
    plt.xlim(zmin,-zmin)
    plt.ylim(0,)
    fig.tight_layout()
    fig.savefig(fig_filename)


def plot_sym_exp_free_energy(z_windows, delta_G, delta_G_err, diff_sym, T, kB=1.9872041e-3, z_units=u'\u00c5',
        fig_filename='expdelG-sym.pdf', grid=True):
    """Plot symmetrized delta G
    
    Params
    ------
    z_windows : np.ndarray, shape=(n,)
        The location of the windows in z
    delta_G : np.ndarray, shape=(n,)
        The symmetrized free energy profile, in energy units
    delta_G_err : np.ndarray, shape=(n,)
        The error in the symmetrized free energy profile, in energy units
    z_units : str
        The units of the z-values in z_windows
    energy_units : str
        The units of delta_G
    fig_filename : str
        The name of the figure file to write
    grid : bool
        Draw gridlines on major ticks if True

    Returns
    -------
    None. This figure draws a figure of the symmetrized free energy profile
    and saves it to disk.
    """

    fig, ax = plt.subplots()
    ax.plot(z_windows, np.exp(delta_G/(kB*T))) # dimensionless
    ax.plot(z_windows, 1/diff_sym) # s/cm2 
    err = np.exp(delta_G) * delta_G_err
    val = np.exp(delta_G)
    #ax.fill_between(z_windows, np.exp(delta_G), 
    #        np.exp(delta_G-delta_G_err),
    #        facecolor='#a8a8a8', edgecolor='#a8a8a8')
    ax.set_xlabel(u'z [{0}]'.format(z_units))
    ax.set_ylabel(u'1/D')
    ax.grid(grid)
    zmin = z_windows[0]    
    plt.xlim(zmin,-zmin)
    fig.tight_layout()
    fig.savefig(fig_filename)
