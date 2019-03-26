# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

import argparse
import sys
import os


# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--n', type = str, required = True, help = 'The experiment number for the run')
parser.add_argument('--data', type=str, default='data',
					help='location of the data corpus. We suggest you change the default\
					here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
					help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
					help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
					help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
					help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
					help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
					help='size of hidden layers. IMPORTANT: for the transformer\
					this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
					help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
					help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
					help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
					help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
					help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
					(dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true') 
parser.add_argument('--save_dir', type=str, default='',
					help='path to save the experimental config, logs, model \
					This is automatically generated based on the command line \
					arguments you pass and only needs to be set if you want a \
					custom dir name')
parser.add_argument('--evaluate', action='store_true',
					help="use this flag to run on the test set. Only do this \
					ONCE for each model setting, and only after you've \
					completed ALL hyperparameter tuning on the validation set.\
					Note we are not requiring you to do this.")
args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]
# print(sys.argv)
# sys.exit()
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:] if '--n=' not in flag]
file_path = os.path.join(args.save_dir+'_'.join([argsdict['model'], argsdict['optimizer']] + flags))
file_path = file_path + '_0'
# Get the file path
experiment_number = args.n

# Define a function to plot graph
def PlotEpochGraphFromFile(file_path, experiment_number) :
	
	# Get the log file contents
	data_path = os.path.join(file_path, 'learning_curves.npy')
	data = np.load(data_path)
	# Get dictionary
	data_dict = data.item()
	# Get the values
	train_ppls = data_dict['train_ppls']
	val_ppls = data_dict['val_ppls']
	train_losses = data_dict['train_losses']
	val_losses = data_dict['val_losses']
	times = data_dict['times']
	# Create arrays
	epochs = np.arange(1, 41)
	
	# Get best validation index
	best_val_ppl_idx = np.argmin(np.array(val_ppls))
	best_val_ppl = val_ppls[int(best_val_ppl_idx)]
	best_val_ppl_idx = best_val_ppl_idx + 1

	# Get the stats for legend
	stats = {}
	config_path = os.path.join(file_path, 'exp_config.txt')
	with open(config_path) as fp:
		lines = fp.readlines()
		for line in lines:
			tokens = line.split()
			if tokens[0] == "emb_size":
				stats["emb"] = tokens[1]
			elif tokens[0] == "optimizer":
				stats["opt"] = tokens[1]
			elif tokens[0] == "initial_lr":
				stats["lr"] = tokens[1]
			elif tokens[0] == "dp_keep_prob":
				stats["dp"] = tokens[1]
			elif tokens[0] == "model":
				stats["model"] = tokens[1]
			# elif tokens[0] == "batch_size":
			# 	stats["batch"] = tokens[1]

	# Create smooth
	epochs_smooth = np.linspace(epochs.min(), epochs.max(), 200)
	train_ppls_smooth = spline(epochs, train_ppls, epochs_smooth)
	val_ppls_smooth = spline(epochs, val_ppls, epochs_smooth)

	legend_str = 'opt=' + stats['opt'] + ' ' + 'lr=' + stats['lr'] + ' ' + 'emb=' + stats['emb'] + ' ' + 'dp=' + stats['dp']

	# Create new graph
	fig = plt.figure()
	# Plot train ppls
	plt.plot(epochs_smooth, train_ppls_smooth, c = 'r', label = legend_str + ' train', linestyle = '-', alpha = 0.65)
	plt.scatter(epochs, train_ppls, c = 'r', marker = 'x', alpha = 0.5)
	# Plot val ppls
	plt.plot(epochs_smooth, val_ppls_smooth, c = 'b', label = legend_str + ' val', linestyle = '--', alpha = 0.65)
	plt.scatter(epochs, val_ppls, c = 'b', marker = 'x', alpha = 0.5)
	# Plot the location of the best model
	plt.scatter(best_val_ppl_idx, best_val_ppl, marker = '*', c = 'purple', s = 100)
	# Title
	plt.title('4.3 ' + stats['model'] + ' experiment ' + experiment_number + ' learning curve')
	# plt.legend(loc = 'upper right')
	plt.legend()
	plt.xlabel(r'Epochs$\rightarrow$')
	plt.ylabel(r'Perplexity$\rightarrow$')
	# plt.ylim((0, 500))
	# Save
	save_path = stats['model'] + ' Experiment ' + experiment_number + ' Learning Curve'
	save_path = save_path + '_Epochs.pdf'
	save_path = save_path.replace(' ', '')
	save_path = save_path.replace('_', '')
	plt.savefig(save_path, dpi = 1000)
	# Show!
	# plt.show()
	plt.close()


# Define a function to plot graph with time
def PlotTimeGraphFromFile(file_path, experiment_number) :
	
	# Get the log file contents
	data_path = os.path.join(file_path, 'learning_curves.npy')
	data = np.load(data_path)
	# Get dictionary
	data_dict = data.item()
	# Get the values
	train_ppls = data_dict['train_ppls']
	val_ppls = data_dict['val_ppls']
	train_losses = data_dict['train_losses']
	val_losses = data_dict['val_losses']
	times = data_dict['times']
	# # Create arrays
	# epochs = np.arange(1, 41)

	# Create cumulative time list
	epochs = []
	epochs.append(times[0])
	for i in range(1, 40) :
		epochs.append(times[i] + epochs[-1])
	epochs = np.array(epochs)
	
	# Get best validation index
	best_val_ppl_idx = np.argmin(np.array(val_ppls))
	best_val_ppl = val_ppls[int(best_val_ppl_idx)]
	best_val_ppl_idx = epochs[best_val_ppl_idx]

	# Get the stats for legend
	stats = {}
	config_path = os.path.join(file_path, 'exp_config.txt')
	with open(config_path) as fp:
		lines = fp.readlines()
		for line in lines:
			tokens = line.split()
			if tokens[0] == "emb_size":
				stats["emb"] = tokens[1]
			elif tokens[0] == "optimizer":
				stats["opt"] = tokens[1]
			elif tokens[0] == "initial_lr":
				stats["lr"] = tokens[1]
			elif tokens[0] == "dp_keep_prob":
				stats["dp"] = tokens[1]
			elif tokens[0] == "model":
				stats["model"] = tokens[1]
			# elif tokens[0] == "batch_size":
			# 	stats["batch"] = tokens[1]

	# Create smooth
	epochs_smooth = np.linspace(epochs.min(), epochs.max(), 200)
	train_ppls_smooth = spline(epochs, train_ppls, epochs_smooth)
	val_ppls_smooth = spline(epochs, val_ppls, epochs_smooth)

	legend_str = 'opt=' + stats['opt'] + ' ' + 'lr=' + stats['lr'] + ' ' + 'emb=' + stats['emb'] + ' ' + 'dp=' + stats['dp']

	# Create new graph
	fig = plt.figure()
	# Plot train ppls
	plt.plot(epochs_smooth, train_ppls_smooth, c = 'r', label = legend_str + ' train', linestyle = '-', alpha = 0.65)
	plt.scatter(epochs, train_ppls, c = 'r', marker = 'x', alpha = 0.5)
	# Plot val ppls
	plt.plot(epochs_smooth, val_ppls_smooth, c = 'b', label = legend_str + ' val', linestyle = '--', alpha = 0.65)
	plt.scatter(epochs, val_ppls, c = 'b', marker = 'x', alpha = 0.5)
	# Plot the location of the best model
	plt.scatter(best_val_ppl_idx, best_val_ppl, marker = '*', c = 'purple', s = 100)
	# Title
	plt.title('4.3 ' + stats['model'] + ' experiment ' + experiment_number + ' learning curve')
	# plt.legend(loc = 'upper right')
	plt.legend()
	plt.xlabel(r'Time $(s) \rightarrow$')
	plt.ylabel(r'Perplexity$\rightarrow$')
	# plt.ylim((0, 500))
	# Save
	save_path = stats['model'] + ' Experiment ' + experiment_number + ' Learning Curve'
	save_path = save_path + '_Timed.pdf'
	save_path = save_path.replace(' ', '')
	save_path = save_path.replace('_', '')
	plt.savefig(save_path, dpi = 1000)
	# Show!
	# plt.show()
	plt.close()

PlotEpochGraphFromFile(file_path, experiment_number)
PlotTimeGraphFromFile(file_path, experiment_number)