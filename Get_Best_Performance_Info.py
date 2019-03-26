# Dependencies
import numpy as np

import argparse
import os
import sys


# Define the argument parser
parser = argparse.ArgumentParser()
# parser.add_argument('--file_path', type = str, required = True, help = 'The path to the folder where all info is stored')
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

flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:] if '--n=' not in flag]
file_path = os.path.join(args.save_dir+'_'.join([argsdict['model'], argsdict['optimizer']] + flags))
file_path = file_path + '_0'
# Add model name
file_path = os.path.join(args.model.lower(), file_path)
# Get the file path
experiment_number = args.n


# Define a function to get the best performance information
def GetBestPerformanceInfo(file_path) :

	"""
	inputs :

	file_path :
		The path where info stored
	"""

	"""
	outputs :

	best_index = best_val_ppl_idx + 1
	best_train_ppl = train_ppls[best_val_ppl_idx]
	best_val_ppl = val_ppls[best_val_ppl_idx]
	best_train_loss = train_losses[best_val_ppl_idx]
	best_val_loss = val_losses[best_val_ppl_idx]	
	"""

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

	# Get best validation index
	best_val_ppl_idx = np.argmin(np.array(val_ppls))
	# Get info
	best_index = best_val_ppl_idx + 1
	best_train_ppl = train_ppls[best_val_ppl_idx]
	best_val_ppl = val_ppls[best_val_ppl_idx]
	best_train_loss = train_losses[best_val_ppl_idx]
	best_val_loss = val_losses[best_val_ppl_idx]

	return best_index, best_train_ppl, best_val_ppl, best_train_loss, best_val_loss


# Call
best_index, best_train_ppl, best_val_ppl, best_train_loss, best_val_loss = GetBestPerformanceInfo(file_path)
# print('Best Index : ' + str(best_index) + ' Best Train PPL : ' + str(best_train_ppl)  +  ' Best Val PPL : ' + str(best_val_ppl)  +  ' Best Train Loss : ' + str(best_train_loss) +  ' Best Val Loss : ' + str(best_val_loss))
print(args.n, ' & ', ' {0:.2f} '.format(best_val_ppl), ' & ', ' {0:.2f} '.format(best_train_ppl))