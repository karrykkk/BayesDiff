# imagenet256
path="imagenet256_uvit_huge.py"
steps="50"
DIS="time_uniform"
mc_size="10"
sample_batch_size="16"
total_n_sample="96"
# the size of dataset used for Laplace approximation: size_of_total_dataset//train_la_data_size
train_la_data_size="50"
fixed_class="11" #all classes are generated if fixed_class="10000" else range from 0 to 999
encoder_path="your_encoder_path"
uvit_path="your_uvit_path"
DEVICES=0
CUDA_VISIBLE_DEVICES=$DEVICES python dpm_solver_skipUQ.py \
--config $path --timesteps=$steps --eta 0 --skip_type=$DIS --train_la_batch_size 16 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --encoder_path=$encoder_path --uvit_path=$uvit_path
