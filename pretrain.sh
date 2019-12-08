python3 bert_pytorch/pretrain.py \
	--dataset="/homes/gws/gezhang/jupyter-notebook-analysis/new_cell_graphs.txt" \
	--output_path='mlm_lg.model' \
	--cuda_devices=0 \
	--log_freq=100 \
	--epochs=200 \
	--layers=4 \
	--attn_heads=4 \
	--lr=0.01 \
	--batch_size=16 \
	--num_workers=1 \
	--duplicate=5 \
	--dropout=0.1