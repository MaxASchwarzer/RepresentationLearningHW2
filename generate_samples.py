from models import RNN, GRU
from models import make_model as TRANSFORMER
import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(
	description='PyTorch Penn Treebank Language Modeling')

parser.add_argument('--data', type=str, default='data',
					help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
					help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--load_path', type=str, default='',
					help='Path to best model')
parser.add_argument('--seq_len', type=int, default=35,
					help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
					help='size of one minibatch')
parser.add_argument('--hidden_size', type=int, default=200,
					help='size of hidden layers. IMPORTANT: for the transformer\
					this must be a multiple of 16.')
parser.add_argument('--num_layers', type=int, default=2,
					help='number of LSTM layers')
# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
					help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
					help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
					help='dropout *keep* probability (dp_keep_prob=0 means no dropout')
parser.add_argument('--top_words', type=int, default=50,
					help='Top number of words to choose from for generating the seed')

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
					help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
print(args)


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS

# Use the GPU if you have one
if torch.cuda.is_available():
	print("Using the GPU")
	device = torch.device("cuda")
else:
	print("WARNING: You are about to run on cpu, and this will likely run out \
	  of memory. \n You can try setting batch_size=1 to reduce memory usage")
	device = torch.device("cpu")



def _read_words(filename):
	with open(filename, "r") as f:
	  return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	id_to_word = dict((v, k) for k, v in word_to_id.items())

	return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
	train_path = os.path.join(data_path, prefix + ".train.txt")
	valid_path = os.path.join(data_path, prefix + ".valid.txt")
	test_path = os.path.join(data_path, prefix + ".test.txt")

	word_to_id, id_2_word = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path, word_to_id)
	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	return train_data, valid_data, test_data, word_to_id, id_2_word


# LOAD DATA
print('Loading data from '+args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# Display utility function
#
###############################################################################
def display_and_save(generated_text, model_arch):
	seq_len = generated_text.shape[0]
	batch_size = generated_text.shape[1]
	list_sentences = []
	for seq_idx in range(batch_size):
		print("Sequence {}".format(seq_idx))
		tokens = generated_text[:, seq_idx].reshape(-1)
		sentence = []
		for token in tokens:
			sentence.append(id_2_word[token])
		sentence = " ".join(sentence)
		print("\n", sentence, "\n")
		list_sentences.append(sentence)
	np.save('generated_sentences_' + str(model_arch) + '_' + str(args.seq_len), np.array(list_sentences))



###############################################################################
#
# Create models
#
###############################################################################
if args.model == 'RNN':
	model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
				seq_len=args.seq_len, batch_size=args.batch_size,
				vocab_size=vocab_size, num_layers=args.num_layers,
				dp_keep_prob=args.dp_keep_prob)
elif args.model == 'GRU':
	model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
				seq_len=args.seq_len, batch_size=args.batch_size,
				vocab_size=vocab_size, num_layers=args.num_layers,
				dp_keep_prob=args.dp_keep_prob)


###############################################################################
#
# Initialize models with parameters
#
###############################################################################

model.load_state_dict(torch.load(args.load_path, map_location=device))
model.eval()
model = model.to(device)


###############################################################################
#
# Generate sentences
#
###############################################################################
CONST_RNN = "RNN"
CONST_GRU = "GRU"

# Generate the seed words in the range 0 to TOP_WORDS
input_ = np.random.randint(0, args.top_words, (args.batch_size, ))
inputs = torch.from_numpy(input_.astype(np.int64)).contiguous().to(device)

print('####### Generating sequences of using ' + args.model + ' ######## \n')
hidden = model.init_hidden()
generated_text = model.generate(inputs, hidden, args.seq_len)
display_and_save(generated_text, args.model)
