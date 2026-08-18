"""Generate test images for the M1 and M2 evaluation runs.

Examples:
    python batch_generate.py \
        --model_path /mnt/scratch/fkwt0359/output_model \
        --test_dir ./hpc_ready_dataset/test \
        --output_dir ./eval_results/M1_generated

    python batch_generate.py \
        --model_path /mnt/scratch/fkwt0359/output_model_ablation \
        --test_dir ./ablation_dataset/test \
        --output_dir ./eval_results/M2_generated
"""

import os
import json
import argparse
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing the trained ControlNet model")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Test split containing images/, depths/, and captions.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for generated images")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    captions_path = os.path.join(args.test_dir, "captions.json")
    with open(captions_path, "r") as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} test captions")

    print(f"Loading ControlNet from {args.model_path}")
    controlnet = ControlNetModel.from_pretrained(args.model_path, torch_dtype=torch.float16)

    print(f"Loading Stable Diffusion pipeline {args.base_model}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("Pipeline ready")

    depths_dir = os.path.join(args.test_dir, "depths")
    depth_files = sorted(os.listdir(depths_dir))
    print(f"Found {len(depth_files)} depth maps")

    generated_count = 0
    skipped_count = 0

    for i, depth_fname in enumerate(depth_files):
        name_no_ext = os.path.splitext(depth_fname)[0]
        caption = None
        for ext in [".jpg", ".png", ".jpeg"]:
            key = name_no_ext + ext
            if key in captions:
                caption = captions[key]
                break

        if caption is None:
            print(f"Skipping {depth_fname}: no matching caption")
            skipped_count += 1
            continue

        depth_path = os.path.join(depths_dir, depth_fname)
        depth_image = Image.open(depth_path).convert("RGB")

        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        result = pipe(
            caption,
            image=depth_image,
            num_inference_steps=args.num_inference_steps,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]

        save_path = os.path.join(args.output_dir, name_no_ext + ".png")
        result.save(save_path)
        generated_count += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(depth_files):
            print(f"Progress: {i + 1}/{len(depth_files)}; generated {generated_count}")

    print(f"\nFinished: generated {generated_count}, skipped {skipped_count}")
    print(f"Output directory: {args.output_dir}")

    del pipe, controlnet
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
