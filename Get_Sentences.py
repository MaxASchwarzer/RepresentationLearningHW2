# Dependencies
import numpy as np


for a_file in ['generated_sentences_RNN_35.npy', 'generated_sentences_RNN_70.npy', 'generated_sentences_GRU_35.npy', 'generated_sentences_GRU_70.npy'] :

	contents = np.load(a_file)
	print('[FILE] : ', a_file)

	for i in range(contents.shape[0]) :
		 print('Sentence ', i, '\n', contents[i], '\n')
