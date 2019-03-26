#!/bin/bash
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 --n=Q.4.1
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 --n=Q.4.2.1 
python Plot_Graph_From_File.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 --n=Q.4.2.2 
# SGD
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 --n=1 
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 --n=2
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 --n=3 
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 --n=4
# SGD_LR_SCHEDULE
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40  --n=5
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=40 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40 --n=6
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40  --n=7
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=40 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40 --n=8
# ADAM
python Plot_Graph_From_File.py --model=GRU --optimizer=ADAM --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40   --n=9
python Plot_Graph_From_File.py --model=GRU --optimizer=ADAM --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.5 --save_best --num_epochs=40  --n=10
python Plot_Graph_From_File.py --model=GRU --optimizer=ADAM --initial_lr=1e-3 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40   --n=11
python Plot_Graph_From_File.py --model=GRU --optimizer=ADAM --initial_lr=1e-4 --batch_size=128 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.9 --save_best --num_epochs=40  --n=12
# SGD-LR Finetune
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --emb_size=256 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 --n=13
python Plot_Graph_From_File.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --emb_size=512 --num_layers=2 --dp_keep_prob=0.35 --save_best --num_epochs=40 --n=14