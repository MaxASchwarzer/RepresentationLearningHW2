# Dependencies
import numpy as np
import matplotlib.pyplot as plt


# Define a function to plot graph from .npz files
def FromNPZ(file_path) :

	data = np.load(file_path)
	losses = data['losses']
	return losses


# Get the .npy files for loss
rnn_loss_path = './RNN_losses.npy'
gru_loss_path = './GRU_losses.npy'
trf_loss_path = './TRF_losses.npz'
# Load data
rnn_loss = np.load(rnn_loss_path)
gru_loss = np.load(gru_loss_path)
trf_loss = FromNPZ(trf_loss_path)

print(rnn_loss.shape)
print(gru_loss.shape)


# Define a fnction to plot the RNN and GRU npy
def PlotRNN(rnn_loss) : 

	fig = plt.figure()

	plt.plot(np.arange(rnn_loss.shape[0]) + 1.0, rnn_loss, c = 'r', linestyle = '--', alpha = 0.75, label = 'Validation Loss for RNN')
	plt.scatter(np.arange(rnn_loss.shape[0]) + 1.0, rnn_loss, c = 'r', marker = 'x')

	plt.xlabel(r'Time-Step $\rightarrow$')
	plt.ylabel(r'Averaged Validation Loss $\rightarrow$')

	plt.legend()

	plt.title(r'The Comparison of Average Validation Loss for Best RNN')
	
	plt.savefig('RNNValidationLossComparison.pdf', dpi = 1000)
	plt.show()
	plt.close()


def PlotGRU(gru_loss) :
	
	fig_ = plt.figure()

	plt.plot(np.arange(gru_loss.shape[0]) + 1.0, gru_loss, c = 'g', linestyle = '--', alpha = 0.75, label = 'Validation Loss for GRU')
	plt.scatter(np.arange(gru_loss.shape[0]) + 1.0, gru_loss, c = 'g', marker = 'x')

	plt.xlabel(r'Time-Step $\rightarrow$')
	plt.ylabel(r'Averaged Validation Loss $\rightarrow$')

	plt.legend()

	plt.title(r'The Comparison of Average Validation Loss for Best GRU')
	
	plt.savefig('GRUValidationLossComparison.pdf', dpi = 1000)
	plt.show()


# Define a function to plot the npy
def PlotFromNPY(rnn_loss, gru_loss, trf_loss) :

	fig = plt.figure()

	plt.plot(np.arange(rnn_loss.shape[0]) + 1.0, rnn_loss, c = 'r', linestyle = '--', alpha = 0.75, label = 'Validation Loss for RNN')
	plt.scatter(np.arange(rnn_loss.shape[0]) + 1.0, rnn_loss, c = 'r', marker = 'x')
	plt.plot(np.arange(gru_loss.shape[0]) + 1.0, gru_loss, c = 'g', linestyle = '--', alpha = 0.75, label = 'Validation Loss for GRU')
	plt.scatter(np.arange(gru_loss.shape[0]) + 1.0, gru_loss, c = 'g', marker = 'x')
	plt.plot(np.arange(trf_loss.shape[0]) + 1.0, trf_loss, c = 'b', linestyle = '--', alpha = 0.75, label = 'Validation Loss for Transformer')
	plt.scatter(np.arange(trf_loss.shape[0]) + 1.0, trf_loss, c = 'b', marker = 'x')

	plt.xlabel(r'Time-Step $\rightarrow$')
	plt.ylabel(r'Averaged Validation Loss $\rightarrow$')

	plt.legend()

	plt.title(r'The Comparison of Average Validation Loss for Different Best Architectures')
	
	plt.savefig('ValidationLossComparison.pdf', dpi = 1000)
	plt.show()


PlotRNN(rnn_loss)
PlotGRU(gru_loss)
PlotFromNPY(rnn_loss, gru_loss, trf_loss)
