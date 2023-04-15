set -euv

# TODO: how to fine tune?
# for training, need to install `diffusers` from local:
# https://huggingface.co/docs/diffusers/installation#install-from-source
#
# pip install -e ".[torch]"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# moved from ~/.cache/huggingface/.....
# export MODEL_NAME='sd-compvis-model'

OUTPUT_DIR="dreambooth_model1"
INSTANCE_DIR="data/instance_images"
CAPTIONS_DIR="data/captions"

# if GPU, set --mixed_precision="fp16"
#
# training param to tune
#  --max_train_steps=15000 \
#  --learning_rate=1e-05 \
#  --use_8bit_adam \
#   --captions_dir="$CAPTIONS_DIR" \

# dump only textenc

accelerate launch --mixed_precision="no" examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400

