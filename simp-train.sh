CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train.py -c ./conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode lif --use-smplified-model