# Cooperate UQ into ddim sampler, experiment results will be saved in ./ddim_exp/skipUQ/
CUDA_VISIBLE_DEVICES=0 python ddim_skipUQ.py --from_file ./prompts.txt \
--ckpt your_local_model_path --local_image_path your_local_image_path --laion_art_path your_laion_art_path \
--H 512 --W 512 --scale 3 --train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# Cooperate UQ into dpm-solver-2 sampler, experiment results will be saved in ./dpm_solver_2_exp/skipUQ/
CUDA_VISIBLE_DEVICES=1 python dpmsolver_skipUQ.py --from_file ./prompts.txt \
--ckpt your_local_model_path --local_image_path your_local_image_path --laion_art_path your_laion_art_path \
--H 512 --W 512 --scale 3 --train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
