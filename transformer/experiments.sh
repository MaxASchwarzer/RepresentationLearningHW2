#!/bin/bash
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=12 --dp_keep_prob=0.8 --save_best --num_epochs 100
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1. --batch_size=64 --seq_len=35 --hidden_size=512 --num_layers=12 --dp_keep_prob=0.9 --save_best --num_epochs 40
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=12 --dp_keep_prob=0.5 --save_best --num_epochs 40
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1. --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=12 --dp_keep_prob=0.5 --save_best --num_epochs 40
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1. --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=12 --dp_keep_prob=0.5 --save_best --num_epochs 100
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1. --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=12 --dp_keep_prob=0.9 --save_best --num_epochs 100
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=12 --dp_keep_prob=0.9 --save_best --num_epochs 40
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=12 --dp_keep_prob=0.9 --save_best --num_epochs 100
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.5 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=1 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9 --save_best
