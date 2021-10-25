for seed in 4885 1234 5678
  do
    CUDA_VISIBLE_DEVICES=3 python3 train.py --data_mode=casia --exp=interproto_casia_$seed --seed=$seed --wandb --tensorboard
  done