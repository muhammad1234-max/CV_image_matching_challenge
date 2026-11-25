#!/bin/bash

CONFIG_NAME="v35"
CONFIG="/kaggle/input/imc25-codes/configs/${CONFIG_NAME}.json"

python /kaggle/input/imc25-codes/main.py \
    --config $CONFIG \
    --dataset ETs \
    --prefix /kaggle/input/image-matching-challenge-2025/train \
    --keep
python /kaggle/input/imc25-codes/evaluate.py \
    --working /kaggle/working/output/${CONFIG_NAME}/ETs/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset amy_gardens \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/amy_gardens/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset stairs \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/stairs/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset fbk_vineyard \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/fbk_vineyard/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_brandenburg_british_buckingham \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_brandenburg_british_buckingham/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_piazzasanmarco_grandplace \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_piazzasanmarco_grandplace/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_sacrecoeur_trevi_tajmahal \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_sacrecoeur_trevi_tajmahal/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_stpeters_stpauls \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_stpeters_stpauls/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset imc2024_lizard_pond \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/imc2024_lizard_pond/


# CONFIG_NAME="v19"
# CONFIG="/kaggle/input/imc25-codes/configs/${CONFIG_NAME}.json"

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset ETs \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/ETs/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset amy_gardens \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/amy_gardens/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset stairs \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/stairs/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset fbk_vineyard \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/fbk_vineyard/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_brandenburg_british_buckingham \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_brandenburg_british_buckingham/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_piazzasanmarco_grandplace \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_piazzasanmarco_grandplace/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_sacrecoeur_trevi_tajmahal \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_sacrecoeur_trevi_tajmahal/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_stpeters_stpauls \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_stpeters_stpauls/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset imc2024_lizard_pond \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/imc2024_lizard_pond/


# CONFIG_NAME="v20"
# CONFIG="/kaggle/input/imc25-codes/configs/${CONFIG_NAME}.json"

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset ETs \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/ETs/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset amy_gardens \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/amy_gardens/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset stairs \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/stairs/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset fbk_vineyard \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/fbk_vineyard/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_brandenburg_british_buckingham \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_brandenburg_british_buckingham/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_piazzasanmarco_grandplace \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_piazzasanmarco_grandplace/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_sacrecoeur_trevi_tajmahal \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_sacrecoeur_trevi_tajmahal/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset pt_stpeters_stpauls \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/pt_stpeters_stpauls/

# python /kaggle/input/imc25-codes/main.py \
#     --config $CONFIG \
#     --dataset imc2024_lizard_pond \
#     --prefix /kaggle/input/image-matching-challenge-2025/train \
#     --keep
# python /kaggle/input/imc25-codes/evaluate.py \
#     --working /kaggle/working/output/${CONFIG_NAME}/imc2024_lizard_pond/
