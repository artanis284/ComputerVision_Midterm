$ErrorActionPreference = "Stop"

python task3_unet_segmentation.py --loss ce --epochs 20 --batch-size 8 --size 128
python task3_unet_segmentation.py --loss dice --epochs 20 --batch-size 8 --size 128
python task3_unet_segmentation.py --loss ce_dice --epochs 20 --batch-size 8 --size 128
