CUDA_VISIBLE_DEVICES=6 python ddim_skipUQ.py --from_file /home///DiffusionUQ/sd/prompts.txt \
--H 512 --W 512 --scale 3 --train_la_data_size 1000 --sample_batch_size 2 \
--train_la_batch_size 10 --total_n_samples 48 --timesteps 50
