# imagenet128
DEVICES="7"
data="imagenet128_guided"
steps="50"
mc_size="10"
sample_batch_size="16"
total_n_sample="96"
# the size of dataset used for Laplace approximation: size_of_total_dataset//train_la_data_size
train_la_data_size="200"
fixed_class="10"
DIS="logSNR"

CUDA_VISIBLE_DEVICES=$DEVICES python dpm_solver_skipUQ.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 16 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --seed=$seed

# CELEBA
# DEVICES="0"
# data="celeba"
# steps="50"
# mc_size="50"
# sample_batch_size="16"
# total_n_sample="96"
# train_la_data_size="50"
# DIS="logSNR"
# fixed_class="0"

# CUDA_VISIBLE_DEVICES=$DEVICES python dpm_solver_skipUQ.py \
# --config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 64 \
# --mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
# --total_n_sample=$total_n_sample --fixed_class=$fixed_class
