# Train AdaIN style-transfer network

# Create Pseudo-Paintings


# train Step1 with each dataset
python train_src.py -cfg configs/train/deeplabv2_r101_src_pascal_realism.yaml OUTPUT_DIR results/step1/pseudo_realism
python train_src.py -cfg configs/train/deeplabv2_r101_src_pascal_impressionism.yaml OUTPUT_DIR results/step1/pseudo_impressionism
python train_src.py -cfg configs/train/deeplabv2_r101_src_pascal_post_impressionism.yaml OUTPUT_DIR results/step1/pseudo_post_impressionism
python train_src.py -cfg configs/train/deeplabv2_r101_src_pascal_expressionism.yaml OUTPUT_DIR results/step1/pseudo_expressionism

# train Step2 with each dataset
python train_adv.py -cfg configs/train/deeplabv2_r101_adv_realism.yaml OUTPUT_DIR results/step2/realism resume results/step1/pseudo_realism/model_iter020000.pth
python train_adv.py -cfg configs/train/deeplabv2_r101_adv_impressionism.yaml OUTPUT_DIR results/step2/impressionism resume results/step1/pseudo_impressionism/model_iter020000.pth
python train_adv.py -cfg configs/train/deeplabv2_r101_adv_post_impressionism.yaml OUTPUT_DIR results/step2/post_impressionism resume results/step1/pseudo_post_impressionism/model_iter020000.pth
python train_adv.py -cfg configs/train/deeplabv2_r101_adv_expressionism.yaml OUTPUT_DIR results/step2/expressionism resume results/step1/pseudo_expressionism/model_iter020000.pth

