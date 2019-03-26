# Dependencies
import numpy as np
import matplotlib.pyplot as plt

import argparse


# Create a parser for gradient arrays
parser = argparse.ArgumentParser(description = 'Q.5.2 Gradient Plots')
parser.add_argument('--rnn_grad_path', type = str, default = './gradient_computation_RNN.npy')
parser.add_argument('--gru_grad_path', type = str, default = './gradient_computation_GRU.npy')
args = parser.parse_args()


# Define a function to plot the two gradient plots, one for RNN and GRU
def PlotRNN(rnn_npy) :

	"""
	inputs :

	rnn_npy :
		The path to .npy file with RNN grads
	gru_npy :
		The path to .npy file with GRU grads
	"""

	"""
	outputs :
	"""

	# Get RNN and GRU data
	rnn_data = np.reshape(np.load(rnn_npy), [-1])

	# Create a BIG figure
	fig = plt.figure(figsize = (10.0, 5.0))

	# Plots : RNN
	plt.plot(1.0 + np.arange(rnn_data.shape[0]), rnn_data, c = 'r', label = 'Gradients for RNN', linestyle = '-', alpha = 0.65)
	plt.scatter(1.0 + np.arange(rnn_data.shape[0]), rnn_data, c = 'r', marker = 'x', alpha = 0.5)
	# # Plots : GRU
	# plt.plot(1.0 + np.arange(gru_data.shape[0]), gru_data, c = 'b', label = 'Gradients for GRU', linestyle = '--', alpha = 0.65)
	# plt.scatter(1.0 + np.arange(gru_data.shape[0]), gru_data, c = 'b', marker = 'x', alpha = 0.5)

	# Decorate
	plt.xlabel(r'Unfolding Time $(t) \rightarrow$')
	plt.ylabel(r'Norm of Gradients $(\nabla_{{\mathbf{h}}_t}\mathcal{L}_T) \rightarrow$')
	plt.title(r'Normalized Averaged Gradients of the Last Time-Step Loss: $\nabla_{{\mathbf{h}}_t}\mathcal{L}_T$ for RNN')
	plt.legend()

	# Save
	plt.savefig('RNNGradientComputation.pdf', dpi = 1000)

	# Show
	# plt.show()


# Define a function to plot the two gradient plots, one for RNN and GRU
def PlotGRU(gru_npy) :

	"""
	inputs :

	rnn_npy :
		The path to .npy file with RNN grads
	gru_npy :
		The path to .npy file with GRU grads
	"""

	"""
	outputs :
	"""

	# Get RNN and GRU data
	gru_data = np.reshape(np.load(gru_npy), [-1])

	# Create a BIG figure
	fig = plt.figure(figsize = (10.0, 5.0))

	# # Plots : RNN
	# plt.plot(1.0 + np.arange(rnn_data.shape[0]), rnn_data, c = 'r', label = 'Gradients for RNN', linestyle = '-', alpha = 0.65)
	# plt.scatter(1.0 + np.arange(rnn_data.shape[0]), rnn_data, c = 'r', marker = 'x', alpha = 0.5)
	# Plots : GRU
	plt.plot(1.0 + np.arange(gru_data.shape[0]), gru_data, c = 'b', label = 'Gradients for GRU', linestyle = '--', alpha = 0.65)
	plt.scatter(1.0 + np.arange(gru_data.shape[0]), gru_data, c = 'b', marker = 'x', alpha = 0.5)

	# Decorate
	plt.xlabel(r'Unfolding Time $(t) \rightarrow$')
	plt.ylabel(r'Norm of Gradients $(\nabla_{{\mathbf{h}}_t}\mathcal{L}_T) \rightarrow$')
	plt.title(r'Normalized Averaged Gradients of the Last Time-Step Loss: $\nabla_{{\mathbf{h}}_t}\mathcal{L}_T$ for GRU')
	plt.legend()

	# Save
	plt.savefig('GRUGradientComputation.pdf', dpi = 1000)

	# Show
	# plt.show()


# Define a function to plot the two gradient plots, one for RNN and GRU
def PlotGradientsFromNPY(rnn_npy, gru_npy) :

	"""
	inputs :

	rnn_npy :
		The path to .npy file with RNN grads
	gru_npy :
		The path to .npy file with GRU grads
	"""

	"""
	outputs :
	"""

	# Get RNN and GRU data
	rnn_data = np.reshape(np.load(rnn_npy), [-1])
	gru_data = np.reshape(np.load(gru_npy), [-1])

	# Create a BIG figure
	fig = plt.figure(figsize = (10.0, 5.0))

	# Plots : RNN
	plt.plot(1.0 + np.arange(rnn_data.shape[0]), rnn_data, c = 'r', label = 'Gradients for RNN', linestyle = '-', alpha = 0.65)
	plt.scatter(1.0 + np.arange(rnn_data.shape[0]), rnn_data, c = 'r', marker = 'x', alpha = 0.5)
	# Plots : GRU
	plt.plot(1.0 + np.arange(gru_data.shape[0]), gru_data, c = 'b', label = 'Gradients for GRU', linestyle = '--', alpha = 0.65)
	plt.scatter(1.0 + np.arange(gru_data.shape[0]), gru_data, c = 'b', marker = 'x', alpha = 0.5)

	# Decorate
	plt.xlabel(r'Unfolding Time $(t) \rightarrow$')
	plt.ylabel(r'Norm of Gradients $(\nabla_{{\mathbf{h}}_t}\mathcal{L}_T) \rightarrow$')
	plt.title(r'Comparison of the Normalized Averaged Gradients of the Last Time-Step Loss: $\nabla_{{\mathbf{h}}_t}\mathcal{L}_T$ for RNN and GRU')
	plt.legend()

	# Save
	plt.savefig('GradientComputationComparison.pdf', dpi = 1000)

	# Show
	# plt.show()


# Plot the graphs
PlotRNN(rnn_npy = args.rnn_grad_path)
PlotGRU(gru_npy = args.gru_grad_path)
PlotGradientsFromNPY(rnn_npy = args.rnn_grad_path, gru_npy = args.gru_grad_path)