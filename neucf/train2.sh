python neucf.py --log_path "./log/normal_ngcfemb/" --epoch_iter 258 --valid_iter 10 --batch_size 4096 --external_embedding True --external_embedding_type 1 --graph_embedding_row_path "./log/ngcf_embedding/row_embedding.npy" --graph_embedding_col_path "./log/ngcf_embedding/col_embedding.npy" --loss_type cross_entropy --decay_step 100000 --layers [256,1024,512,256,128] --reg_layers [0.0001,0.0001,0.0001,0.0001,0.0001]
python neucf.py --log_path "./log/normal_ngcfemb_decay1500/" --epoch_iter 258 --valid_iter 10 --batch_size 4096 --external_embedding True --external_embedding_type 1 --graph_embedding_row_path "./log/ngcf_embedding/row_embedding.npy" --graph_embedding_col_path "./log/ngcf_embedding/col_embedding.npy" --loss_type cross_entropy --decay_step 1500 --layers [256,1024,512,256,128] --reg_layers [0.0001,0.0001,0.0001,0.0001,0.0001]
python neucf.py --log_path "./log/normal_ngcfemb_decay2500/" --epoch_iter 258 --valid_iter 10 --batch_size 4096 --external_embedding True --external_embedding_type 1 --graph_embedding_row_path "./log/ngcf_embedding/row_embedding.npy" --graph_embedding_col_path "./log/ngcf_embedding/col_embedding.npy" --loss_type cross_entropy --decay_step 2500 --layers [256,1024,512,256,128] --reg_layers [0.0001,0.0001,0.0001,0.0001,0.0001]
python neucf.py --log_path "./log/normal_ngcfemb_decay3500/" --epoch_iter 258 --valid_iter 10 --batch_size 4096 --external_embedding True --external_embedding_type 1 --graph_embedding_row_path "./log/ngcf_embedding/row_embedding.npy" --graph_embedding_col_path "./log/ngcf_embedding/col_embedding.npy" --loss_type cross_entropy --decay_step 3500 --layers [256,1024,512,256,128] --reg_layers [0.0001,0.0001,0.0001,0.0001,0.0001]
python neucf.py --log_path "./log/basenn_8/" --model "NeuCF" --epoch_iter 258 --valid_iter 10 --batch_size 4096 --loss_type mse --num_factors 8 --decay_step 2500 --layers [64,32,16,8] --reg_mf 0.0001 --reg_layers [0.0001,0.0001,0.0001,0.0001]
python neucf.py --log_path "./log/basenn_16/" --model "NeuCF" --epoch_iter 258 --valid_iter 10 --batch_size 4096 --loss_type mse --num_factors 16 --decay_step 2500 --layers [128,64,32,16] --reg_mf 0.0001 --reg_layers [0.0001,0.0001,0.0001,0.0001]
