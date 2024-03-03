# ddimUQ.py's IMAGENET128
DEVICES="0"
data="imagenet128_guided"
steps="50"
mc_size="10"
sample_batch_size="16"
total_n_sample="96"
train_la_data_size="200"
DIS="uniform"
fixed_class="10"
seed=1234

CUDA_VISIBLE_DEVICES=$DEVICES python ddim_skipUQ.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size 32 \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --fixed_class=$fixed_class --seed=$seed
