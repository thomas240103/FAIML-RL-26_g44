conda activate rl_env

tensorboard --logdir /Users/luciobaiocchi/projects/FAIML-RL-26_g44/part2/tb_logs/


# SAC + HER
python train_sb3_sac.py --sampling-strategy none --env-type source --timesteps 100000 --her


python train_sb3_sac.py \
--sampling-strategy none \
--env-type source \
--timesteps 100000 \
--her \
--gradient-steps 4 \
--batch-size 512 \
--learning-rate 1e-3 \
--device mps


# train_sb3.py
python part2/train_sb3.py \
--timesteps 500000 \
--algorithm sac \
--learning-rate 0.001 \
--batch-size 2048 \
--gradient-steps -1 \
--learning-starts 100000 \
--buffer-size 1000000 \
--tau 0.05 \
--seed 42 \
--gamma 0.95 \
--save 


python train_sb3.py \
--timesteps 500000 \
--algorithm sac \
--sampling-strategy adr \
--learning-rate 0.001 \
--batch-size 2048 \
--gradient-steps -1 \
--learning-starts 100000 \
--buffer-size 1000000 \
--tau 0.05 \
--seed 42 \
--gamma 0.95 \
--save 

Il modello generato da questo comando sta qui: models/SAC_none_source_500k/model.zip
e il log del training è questo: tb_logs/SAC_none_source_500k_2