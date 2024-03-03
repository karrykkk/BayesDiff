# for imagenet dataset
DEVICES="0"
data="imagenet128_guided"
steps="250"
mc_size="10"
sample_batch_size="16"
total_n_sample="96"
# the size of dataset used for Laplace approximation: size_of_total_dataset//train_la_data_size
train_la_data_size="50"
DIS="uniform"
fixed_class="0"

CUDA_VISIBLE_DEVICES=$DEVICES python ddpm_skipUQ.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 32 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --fixed_class=$fixed_class --seed=1234

# for CELEBA dataset
# DEVICES="0"
# data="celeba"
# steps="250"
# mc_size="10"
# sample_batch_size="16"
# total_n_sample="96"
# # the size of dataset used for Laplace approximation: size_of_total_dataset//train_la_data_size
# train_la_data_size="50"
# DIS="uniform"
# fixed_class="0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ddpm_skipUQ.py \
# --config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 32 \
# --mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
# --total_n_sample=$total_n_sample --fixed_class=$fixed_class
