Here we document the commands used to finetune the ImageNet-pretrained DINO model on other datasets.

CIFAR100

```bash
    python run_with_submitit.py \
    --arch vit_base \
    --patch_size 16 \
    --out_dim 65536 \
    --norm_last_layer true \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 50 \
    --use_fp16 false \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 0.3 \
    --epochs 200 \
    --freeze_last_layer 3 \
    --lr 0.00075 \
    --warmup_epochs 10 \
    --min_lr 2e-6 \
    --global_crops_scale 0.25 1.0 \
    --local_crops_scale 0.05 0.25 \
    --local_crops_number 10 \
    --seed 0 \
    --num_workers 16 \
    --optimizer adamw \
    --momentum_teacher 0.996 \
    --use_bn_in_head false \
    --drop_path_rate 0.1 \
    --grad_from_block 11 \
    --dataset CIFAR100 \
    --data_path /hpi/fs00/share/fg-meinel/datasets/CIFAR100/ \
    --output_dir /hpi/fs00/home/jona.otholt/dino/finetuning/CIFAR100/ \
    --batch_size_per_gpu 256 \
    --ngpus 2 \
    --nodes 1 \
    --account meinel-mlai \
    --partition sorcery \
    --cpus_per_task 20 \
    --constraint 'ARCH:PPC'
```

CUB

```bash
    python run_with_submitit.py \
    --arch vit_base \
    --patch_size 16 \
    --out_dim 65536 \
    --norm_last_layer true \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 50 \
    --use_fp16 false \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 0.3 \
    --epochs 200 \
    --freeze_last_layer 3 \
    --lr 0.00075 \
    --warmup_epochs 10 \
    --min_lr 2e-6 \
    --global_crops_scale 0.25 1.0 \
    --local_crops_scale 0.05 0.25 \
    --local_crops_number 10 \
    --seed 0 \
    --num_workers 16 \
    --optimizer adamw \
    --momentum_teacher 0.996 \
    --use_bn_in_head false \
    --drop_path_rate 0.1 \
    --grad_from_block 11 \
    --dataset CUB200 \
    --data_path /hpi/fs00/share/fg-meinel/datasets/GCD-datasets/cub/ \
    --output_dir /hpi/fs00/home/jona.otholt/dino/finetuning/CUB200/ \
    --batch_size_per_gpu 256 \
    --ngpus 2 \
    --nodes 1 \
    --account meinel-mlai \
    --partition sorcery \
    --cpus_per_task 20 \
    --constraint 'ARCH:PPC'
```
