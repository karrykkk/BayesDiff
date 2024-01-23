# ddpmUQ.py's CELEBA skip
DEVICES="0"
data="celeba"
steps="250"
mc_size="10"
sample_batch_size="30"
total_n_sample="60000"
train_la_data_size="50"
DIS="uniform"
fixed_class="0"

CUDA_VISIBLE_DEVICES=$DEVICES python ddpm_skipUQ.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 32 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --fixed_class=$fixed_class