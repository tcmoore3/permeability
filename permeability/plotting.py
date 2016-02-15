import matplotlib.pyplot as plt
import numpy as np


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
    ax.set_xlabel(u'z, {z_units}'.format(**locals()))
    ax.set_ylabel(u'F(z), {force_units}'.format(**locals()))
    ax.grid(grid)
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
    ax.set_xlabel(u'z, {z_units}'.format(**locals()))
    ax.set_ylabel(u'\u0394G(z), {energy_units}'.format(**locals()))
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)
    plt.show()

def plot_force_acfs_time(time, int_facfs, time_units='ps', grid=True,
        fig_filename='force_acf_per_window.pdf'):
    fig, ax = plt.subplots()
    for int_facf in int_facfs:
        ax.plot(time, int_facf)
    ax.set_xlabel('t, {0}'.format(time_units))
    ax.set_ylabel(r"$\int_0^t$ FACF dt'")
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_diffusion_coefficient_z(z_windows, diffusion_coefficients, 
        z_units=u'\u00c5', D_units=u'1E5 cm\u00b2/s', fig_filename='d_z.pdf',
        grid=True):
    """Plot the diffusion coefficient as a function of z-position.
    """
    fig, ax = plt.subplots()
    ax.plot(z_windows, diffusion_coefficients)
    ax.set_xlabel(u'z, {0}'.format(z_units))
    ax.set_ylabel(u'D, {0}'.format(D_units))
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)

def plot_symmetrized_free_energy(z_windows, delta_G, z_units=u'\u00c5',
        energy_units=u'kcal-mol', fig_filename='delG-sym.pdf', grid=True):
    """Plot symmetrized delta G
    
    Params
    ------
    z_windows : np.ndarray, shape=(n,)
        The location of the windows in z
    delta_G : np.ndarray, shape=(n,)
        The symmetrized free energy profile, in energy units
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
    ax.set_xlabel(u'z, {0}'.format(z_units))
    ax.set_ylabel(u'G(z), {0}'.format(energy_units))
    ax.grid(grid)
    fig.tight_layout()
    fig.savefig(fig_filename)
