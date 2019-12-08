python3 bert_pytorch/__main__.py \
	--dataset="/homes/gws/gezhang/jupyter-notebook-analysis/graphs/cell_with_func.txt" \
	--output_path='same_dataset.model' \
	--cuda_devices=0 \
	--log_freq=1000 \
	--epochs=200 \
	--layers=2 \
	--attn_heads=2 \
	--lr=0.001 \
	--batch_size=32 \
	--num_workers=1 \
	--duplicate=5 \
	--dropout=0 \
	--min_occur=3 \
	--weak_supervise \
	--use_sub_token \
	--seq_len=64
	# --n_topics=10
