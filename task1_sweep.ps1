$ErrorActionPreference = "Stop"

# Task 1 hyper-parameter examples. Edit epochs/batch-size for your GPU budget.
python task1_pet_classification.py --pretrained --arch resnet34 --epochs 10 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-4
python task1_pet_classification.py --pretrained --arch resnet34 --epochs 10 --batch-size 32 --lr-head 3e-4 --lr-backbone 3e-5
python task1_pet_classification.py --pretrained --arch resnet34 --attention se --epochs 10 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-4
python task1_pet_classification.py --random-init --arch resnet34 --epochs 10 --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-3
