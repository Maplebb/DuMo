from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from dumo import DuMo
from src.engine import train_util
from src.configs import config
from typing import Literal
import torch
from PIL import Image
import pandas as pd
import argparse
import numpy as np

def generate_images(model_path, prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0, till_case=1000000, base='1.4'):
    
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    
    if base == '1.4':
        model_version = "../out_checkpoint/SDv1.4_original"
    else:
        print("Base version not supported")
        return 0
    vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder")

    unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet")
    dumo = DuMo.from_pretrained(model_path)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    dumo.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)
    # prompts = df.prompt
    # seeds = df.evaluation_seed

    folder_path = f'{save_path}'
    os.makedirs(folder_path, exist_ok=True)

    for case_idx, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.sd_seed
        case_number = case_idx
        if not (case_number>=from_case and case_number<=till_case):
            continue
        
        if os.path.exists(f"{folder_path}/{case_number}_0.png"):
            print("Image exists: ", f"{folder_path}/{case_number}_0.png")
            continue

        height = row.sd_image_height                        # default height of Stable Diffusion
        width = row.sd_image_width                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = row.sd_guidance_scale            # Scale for classifier-free guidance

        generator = torch.cuda.manual_seed(seed)    # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,device=torch_device
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                (down_block_res_samples, mid_block_res_sample) = dumo(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,)
                noise_pred = unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")
            print("Image saved to :", f"{folder_path}/{case_number}_{num}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_path', help='path of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, default='benchmark/i2p_benchmark.csv')
    parser.add_argument('--save_path', help='folder where to save images', type=str, default="images")
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--base', help='version of stable diffusion to use', type=str, required=False, default='1.4')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()
    
    model_path = args.model_path
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    till_case = args.till_case
    base = args.base
    generate_images(model_path=model_path, prompts_path=prompts_path, save_path=save_path, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, till_case=till_case, base=base)