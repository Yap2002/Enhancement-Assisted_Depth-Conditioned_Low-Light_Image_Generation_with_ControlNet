from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import os
import gc

depth_image_path = "./hpc_ready_dataset/val/depths/Chair_2015_04229.png"
depth_image = Image.open(depth_image_path).convert("RGB")
# prompt = "A picture of a room with a bike and a bed"
# prompt = "A bicycle, a tree and a bench on a cobblestone street"
# prompt = "A bicycle parked in the snow outside of a house at night under a low-light environment"
# prompt = "A picture of a man in a boat at sunset under a low-light environment"
prompt = "A low-light photograph of a wooden chair sitting on dark green grass at night, with a single warm yellow spotlight from the right of the chair"
# prompt = "A picture of a dog walking on a road at night under a low-light environment"
# prompt = "A picture of a table on a deck at night"
# prompt = "A picture of a table on a deck at night under a low-light environment"
# prompt = "A picture of a man standing in front of a windmill at night"
#prompt = "A man standing in front of the highly detailed red Moulin Rouge windmill and building at night, with vibrant colorful neon signs and lettering, complex city street night lighting, high contrast photography."

output_dir = "result_images"
os.makedirs(output_dir, exist_ok=True)

checkpoints = ["2000", "4000", "6000", "8000", "10000"]
base_model_path = "runwayml/stable-diffusion-v1-5"

print("Testing checkpoints...")

for ckpt in checkpoints:
    model_path = f"./my_scratch_models/checkpoint-{ckpt}/controlnet"
    
    print(f"\nLoading checkpoint {ckpt}")
    
    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_attention_slicing() 
    
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(
        prompt,
        image=depth_image,
        num_inference_steps=30,          
        controlnet_conditioning_scale=1.0, 
        guidance_scale=5.0,              
        generator=generator
    ).images[0]
    
    output_name = os.path.join(output_dir, f"test_result_Chair_2015_04229_{ckpt}.png")
    image.save(output_name)
    print(f"Saved {output_name}")
    
    del pipe
    del controlnet
    torch.cuda.empty_cache()
    gc.collect()

print("\nFinished generating images for all five checkpoints.")
