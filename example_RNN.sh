#!/bin/bash
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 
# SGD
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40
# SGD_LR_SCHEDULE
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=40 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=40 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40
# ADAM
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40

# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.01 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.8 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1 --batch_size=64 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40
# python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=.9 --save_best --num_epochs=40

# python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 
# python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=.9 --save_best --num_epochs=40
# python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=4 --dp_keep_prob=.9 --save_best --num_epochs=40

# python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40

