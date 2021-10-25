for seed in 4885 1234 5678
  do
    CUDA_VISIBLE_DEVICES=0 python3 train.py --data_mode=ms1m --exp=interproto_ms1m_$seed --seed=$seed --wandb --tensorboard
  done