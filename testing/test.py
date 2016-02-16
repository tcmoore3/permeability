import permeability as prm
import matplotlib.pyplot as plt
import cPickle as pickle


data_dir = '/Users/tcmoore3/Desktop/sandbox/permeability/data'
#prm.analyze_sweeps(data_dir, verbosity=2)
output = prm.analyze_force_acf_data(data_dir, 305.0, verbosity=1, n_sweeps=None)
pickle.dump(output, open('output.p', 'wb'))
#output = pickle.load(open('output.p', 'rb'))
prm.plot_forces(output['z'], output['forces'], fig_filename='forces.pdf')
prm.plot_free_energy_z(output['z'], output['dG'], fig_filename='delta_G.pdf')
prm.plot_force_acfs_time(output['time'], output['int_facf_windows'])
prm.plot_diffusion_coefficient_z(output['z'], output['d_z'], output['d_z_err'])
prm.plot_symmetrized_free_energy(output['z'], output['dG_sym'], 
        output['dG_sym_err'])
prm.plot_sym_diffusion_coefficient_z(output['z'], output['d_z_sym'], 
        output['d_z_sym_err'])
prm.plot_resistance_z(output['z'], output['R_z'])
import os; os.system('open *.pdf')
