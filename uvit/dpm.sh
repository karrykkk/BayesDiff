# imagenet256
DEVICES="7"
steps="50"
mc_size="10"
sample_batch_size="16"
total_n_sample="96"
# the size of dataset used for Laplace approximation: size_of_total_dataset//train_la_data_size
train_la_data_size="10"
fixed_class="10"
DIS="time_uniform"
encoder_path="your_encoder_path"
uvit_path="your_uvit_path"

CUDA_VISIBLE_DEVICES=$DEVICES python dpm_solver_skipUQ.py \
--timesteps=$steps --skip_type=$DIS --train_la_batch_size 16 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --seed=$seed --encoder_path=$encoder_path --uvit_path=$uvit_path
