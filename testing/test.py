import permeability as prm
import matplotlib.pyplot as plt
import cPickle as pickle


data_dir = '/Users/tcmoore3/Desktop/sandbox/permeability/data'
prm.analyze_sweeps(data_dir, verbosity=2)
output = prm.analyze_force_acf_data(data_dir, 305.0, verbosity=1)
pickle.dump(output, open('output.p', 'wb'))
#output = pickle.load(open('output.p', 'rb'))
prm.plot_forces(output[0], output[2], fig_filename='forces.pdf')
prm.plot_free_energy_z(output[0], output[3], fig_filename='delta_G.pdf')
prm.plot_force_acfs_time(output[1], output[4])
prm.plot_diffusion_coefficient_z(output[0], output[7])
prm.plot_symmetrized_free_energy(output[0], output[8])
