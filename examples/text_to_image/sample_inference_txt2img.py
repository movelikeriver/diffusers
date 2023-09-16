"""
usage:
python sample_inference_txt2img.py

sample code from https://github.com/huggingface/diffusers

one time setup:

conda create -n sd2 pytorch==1.12.1 torchvision==0.13.1
conda activate sd2
# conda install -c conda-forge diffusers==0.12.1    <-- conda version
# conda install -c conda-forge transformers==4.19.2   <-- one repo needs this version
conda install -c conda-forge transformers==4.27.4
conda install -c conda-forge accelerate==0.18.0
conda install -c conda-forge datasets==2.11.0
conda install -c conda-forge ftfy==6.1.1
pip install invisible-watermark


if with pip only,
python3 -m venv difu
source difu/bin/activate
pip3 install --upgrade pip==23.1.2
pip3 install -e ".[torch]" --target=/usr/local/lib/python3.7/dist-packages


pip==23.1.2
accelerate==0.16.0
datasets==2.13.1
ftfy==6.1.1
Jinja2==3.0.3
tensorboard==2.11.2
torchvision==0.13.1
transformers==4.30.2


for training, need to install `diffusers` from local:
https://huggingface.co/docs/diffusers/installation#install-from-source

"""

import time
import torch
from diffusers import StableDiffusionPipeline


load_from_local = True

if not load_from_local:
    # option-1: download from Hub
    # will download to ~/.cache/huggingface/...
    # model_path = 'runwayml/stable-diffusion-v1-5'
    model_path = 'gsdf/Counterfeit-V2.5'
    # model_path = '~/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819'
    # model_path = 'CompVis/stable-diffusion-v1-4'
    # model_path = '~/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/249dd2d739844dea6a0bc7fc27b3c1d014720b28'
    # model_path = 'sd-compvis-model'  # moved from ~/.cache/huggingface/...
    print(f"downloading {model_path}")

else:
    # option-2: load from local path
    # model_path = 'sd-pokemon-model'
    model_path = '../Counterfeit-V2.5'
    print(f"loading from local path {model_path}")

start = time.time()
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
pipe = pipe.to("cuda")
# Recommended if your computer has < 64 GB of RAM
# pipe.enable_attention_slicing()

# Note: maximum sequence length for this model
# prompt = "yoda"
# prompt = "This Elegant 14K Solid Two Tone Gold Mens Wedding Band is 6mm wide.  Center of the Ring has a Satin Finished and edges are Shiny Finish. This Ring is comfort Fitted.\n\n Manufactured in New York, USA. Available in different Metals, Widths, Colors and Finishing."
# prompt = "beautiful elven woman sitting in a white elven city, (full body), (blush), (sitting on stone staircase), pinup pose, (world of warcraft blood elf), (cosplay wig), (medium blonde hair:1.3), (light blue eyes:1.2), ((red, and gold elf minidress)), intricate elven dress"

# https://huggingface.co/gsdf/Counterfeit-V2.5/blob/main/README.md
prompt = "((masterpiece,best quality)),1girl, from below, solo, school uniform, serafuku, sky, cloud, black hair, skirt, sailor collar, looking at viewer, short hair, building, bangs, neckerchief, long sleeves, cloudy sky, power lines, shirt, cityscape, pleated skirt, scenery, blunt bangs, city, night, black sailor collar, closed mouth, black skirt, medium hair, school bag , holding bag"

print(f"=== prompt ===\n{prompt}\n===========\n")

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)


# Results match those from the CPU device after the warmup pass.
img_list = pipe(prompt, num_inference_steps=40, guidance_scale=10, num_images_per_prompt=4).images

print(len(img_list))

for i in range(0, len(img_list)):
    image = img_list[i]
    output_fn = f"../out/output{i}.png"
    print(f"after {(time.time() - start) / 60.0 :.2f} minutes, saving file into {output_fn}")
    image.save(output_fn)
