from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import operator
import os
from functools import reduce
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# import wandb

import sys
sys.path.append('.')
sys.path.append('..')
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.util import instantiate_from_config
import random
import argparse
import time

from constants.const import theme_available, class_available

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    # global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, previous_ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model_prev = load_model_from_config(config_path, previous_ckpt_path, devices[2])
    sampler_prev = DDIMSampler(model_prev)

    model = load_model_from_config(config_path, previous_ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model_prev, sampler_prev, model, sampler

def get_first_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)


    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def get_single_models(config_path, ckpt_path, devices):
    # model_orig = load_model_from_config(config_path, ckpt_path, devices[1])

    model = load_model_from_config(config_path, previous_ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model, sampler

def train_esd(prompt, old_texts, new_texts, prev_forget_texts, retain_texts, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, previous_ckpt_path, devices, output_name, seperator=None, image_size=512, ddim_steps=50, eta=1):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model_prev, sampler_prev, model, sampler = get_models(config_path, ckpt_path, previous_ckpt_path, devices)


    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                # print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                # print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                # print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            # print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                # print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    # print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    # print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    # name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        old_text = random.sample(old_texts,1)[0]
        new_text = random.sample(new_texts,1)[0]
        # prev_forget_text = random.sample(prev_forget_texts,1)[0]
        # retain_text = random.sample(retain_texts,1)[0]
        word = random.sample(words,1)[0]
        # get text embeddings for unconditional and conditional prompts
        old_text_emb = model.get_learned_conditioning([old_text])
        new_text_emb = model.get_learned_conditioning([new_text])
        # prev_forget_text_emb = model.get_learned_conditioning([prev_forget_text])
        # retain_text_emb = model.get_learned_conditioning([retain_text])

        optimizer.zero_grad()

        t_ddim = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_ddim)/ddim_steps)*1000)
        og_num_lim = round((int(t_ddim+1)/ddim_steps)*1000)

        t_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z1 = quick_sample_till_t(old_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            # z2 = quick_sample_till_t(prev_forget_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            # z3 = quick_sample_till_t(retain_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_new = model_orig.apply_model(z1.to(devices[1]), t_ddpm.to(devices[1]), new_text_emb.to(devices[1]))
            e_new2 = model_orig.apply_model(z1.to(devices[1]), t_ddpm.to(devices[1]), old_text_emb.to(devices[1]))
            # e_prev_forget = model_prev.apply_model(z2.to(devices[2]), t_ddpm.to(devices[2]), prev_forget_text_emb.to(devices[2]))
            # e_retain = model_orig.apply_model(z3.to(devices[1]), t_ddpm.to(devices[1]), retain_text_emb.to(devices[1]))
            # e_new = model_orig.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), new_text_emb.to(devices[0]))
            # e_prev_forget = model_prev.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), prev_forget_text_emb.to(devices[0]))
            # e_retain = model_orig.apply_model(z3.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        # breakpoint()
        # get conditional score from ESD model
        e_old = model.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), old_text_emb.to(devices[0]))
        # e_prev_forget_now = model.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), prev_forget_text_emb.to(devices[0]))
        # e_retain_now = model.apply_model(z3.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        e_new.requires_grad = False
        e_new2.requires_grad = False
        # e_prev_forget.requires_grad = False
        # e_retain.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss1 = criteria(e_old.to(devices[0]), e_new.to(devices[0]) - (negative_guidance*(e_new2.to(devices[0]) - e_new.to(devices[0])))) 
        # loss2 = criteria(e_prev_forget.to(devices[0]), e_prev_forget_now.to(devices[0])) 
        # loss3 = criteria(e_retain.to(devices[0]), e_retain_now.to(devices[0])) 
        loss = loss1
        # update weights to erase the concept
        loss.backward()
        # wandb.log({"loss": loss.item()})
        pbar.set_postfix({"loss": loss.item()})
        optimizer.step()


        # 2step
        # 2step
        # 2step
        optimizer.zero_grad()

        prev_forget_text = random.sample(prev_forget_texts,1)[0]
        retain_text = random.sample(retain_texts,1)[0]

        prev_forget_text_emb = model.get_learned_conditioning([prev_forget_text])
        retain_text_emb = model.get_learned_conditioning([retain_text])

        t_ddim1 = torch.randint(ddim_steps, (1,), device=devices[0])
        t_ddim2 = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num1 = round((int(t_ddim1)/ddim_steps)*1000)
        og_num_lim1 = round((int(t_ddim1+1)/ddim_steps)*1000)
        og_num2 = round((int(t_ddim2)/ddim_steps)*1000)
        og_num_lim2 = round((int(t_ddim2+1)/ddim_steps)*1000)

        t_ddpm1 = torch.randint(og_num1, og_num_lim1, (1,), device=devices[0])
        t_ddpm2 = torch.randint(og_num2, og_num_lim2, (1,), device=devices[0])
        

        start_code1 = torch.randn((1, 4, 64, 64)).to(devices[0])
        start_code2 = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept from ESD model
            # z1 = quick_sample_till_t(old_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            z2 = quick_sample_till_t(prev_forget_text_emb.to(devices[0]), start_guidance, start_code1, int(t_ddim1))
            z3 = quick_sample_till_t(retain_text_emb.to(devices[0]), start_guidance, start_code2, int(t_ddim2))
            # get conditional and unconditional scores from frozen model at time step t and image z
            # e_new = model_orig.apply_model(z1.to(devices[1]), t_ddpm.to(devices[1]), new_text_emb.to(devices[1]))
            e_prev_forget = model_prev.apply_model(z2.to(devices[2]), t_ddpm1.to(devices[2]), prev_forget_text_emb.to(devices[2]))
            e_retain = model_orig.apply_model(z3.to(devices[1]), t_ddpm2.to(devices[1]), retain_text_emb.to(devices[1]))
            # e_new = model_orig.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), new_text_emb.to(devices[0]))
            # e_prev_forget = model_prev.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), prev_forget_text_emb.to(devices[0]))
            # e_retain = model_orig.apply_model(z3.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        # breakpoint()
        # get conditional score from ESD model
        # e_old = model.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), old_text_emb.to(devices[0]))
        e_prev_forget_now = model.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), prev_forget_text_emb.to(devices[0]))
        e_retain_now = model.apply_model(z3.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        # e_new.requires_grad = False
        e_prev_forget.requires_grad = False
        e_retain.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        # loss1 = criteria(e_old.to(devices[0]), e_new.to(devices[0])) 
        loss2 = criteria(e_prev_forget.to(devices[0]), e_prev_forget_now.to(devices[0])) 
        loss3 = criteria(e_retain.to(devices[0]), e_retain_now.to(devices[0])) 
        loss = (loss2 + loss3)/2
        # update weights to erase the concept
        loss.backward()
        # wandb.log({"loss": loss.item()})
        pbar.set_postfix({"loss": loss.item()})
        optimizer.step()

    model.eval()
    torch.save({"state_dict": model.state_dict()}, output_name)

def train_first(prompt, old_texts, new_texts, prev_forget_texts, retain_texts, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, previous_ckpt_path, devices, output_name, seperator=None, image_size=512, ddim_steps=50, eta=1):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)
    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_first_models(config_path, ckpt_path, devices)


    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                # print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                # print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                # print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            # print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                # print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    # print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    # print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    # name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        old_text = random.sample(old_texts,1)[0]
        new_text = random.sample(new_texts,1)[0]
        # retain_text = random.sample(retain_texts,1)[0]
        word = random.sample(words,1)[0]
        # get text embeddings for unconditional and conditional prompts
        old_text_emb = model.get_learned_conditioning([old_text])
        new_text_emb = model.get_learned_conditioning([new_text])
        # retain_text_emb = model.get_learned_conditioning([retain_text])

        optimizer.zero_grad()

        t_ddim = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_ddim)/ddim_steps)*1000)
        og_num_lim = round((int(t_ddim+1)/ddim_steps)*1000)

        t_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z1 = quick_sample_till_t(old_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            # z2 = quick_sample_till_t(retain_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            # get conditional and unconditional scores from frozen model at time step t and image z
            # e_new = model_orig.apply_model(z1.to(devices[1]), t_ddpm.to(devices[1]), new_text.to(devices[1]))
            # e_prev_forget = model_prev.apply_model(z2.to(devices[2]), t_ddpm.to(devices[2]), prev_forget_text.to(devices[2]))
            # e_retain = model_orig.apply_model(z3.to(devices[1]), t_ddpm.to(devices[1]), retain_text_emb.to(devices[1]))
            e_new = model_orig.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), new_text_emb.to(devices[0]))
            e_new2 = model_orig.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), old_text_emb.to(devices[0]))
            # e_retain = model_orig.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        # breakpoint()
        # get conditional score from ESD model
        e_old = model.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), old_text_emb.to(devices[0]))
        # e_retain_now = model.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        e_new.requires_grad = False
        e_new2.requires_grad = False
        # e_retain.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss1 = criteria(e_old.to(devices[0]), e_new.to(devices[0]) - (negative_guidance*(e_new2.to(devices[0]) - e_new.to(devices[0])))) 
        # loss3 = criteria(e_retain.to(devices[0]), e_retain_now.to(devices[0])) 
        # loss = eta*loss1 + loss3
        loss = loss1
        # update weights to erase the concept
        loss.backward()
        # wandb.log({"loss": loss.item()})
        pbar.set_postfix({"loss": loss.item()})
        optimizer.step()

        # 2step
        # 2step
        # 2step
        optimizer.zero_grad()

        retain_text = random.sample(retain_texts,1)[0]
        retain_text_emb = model.get_learned_conditioning([retain_text])

        t_ddim = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_ddim)/ddim_steps)*1000)
        og_num_lim = round((int(t_ddim+1)/ddim_steps)*1000)

        t_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept from ESD model
            # z1 = quick_sample_till_t(old_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            z2 = quick_sample_till_t(retain_text_emb.to(devices[0]), start_guidance, start_code, int(t_ddim))
            # get conditional and unconditional scores from frozen model at time step t and image z
            # e_new = model_orig.apply_model(z1.to(devices[1]), t_ddpm.to(devices[1]), new_text.to(devices[1]))
            # e_prev_forget = model_prev.apply_model(z2.to(devices[2]), t_ddpm.to(devices[2]), prev_forget_text.to(devices[2]))
            # e_retain = model_orig.apply_model(z3.to(devices[1]), t_ddpm.to(devices[1]), retain_text_emb.to(devices[1]))
            # e_new = model_orig.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), new_text_emb.to(devices[0]))
            # e_new2 = model_orig.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), old_text_emb.to(devices[0]))
            e_retain = model_orig.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        # breakpoint()
        # get conditional score from ESD model
        # e_old = model.apply_model(z1.to(devices[0]), t_ddpm.to(devices[0]), old_text_emb.to(devices[0]))
        e_retain_now = model.apply_model(z2.to(devices[0]), t_ddpm.to(devices[0]), retain_text_emb.to(devices[0]))
        # e_new.requires_grad = False
        # e_new2.requires_grad = False
        e_retain.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        # loss1 = criteria(e_old.to(devices[0]), e_new.to(devices[0]) - (negative_guidance*(e_new2.to(devices[0]) - e_new.to(devices[0])))) 
        loss3 = criteria(e_retain.to(devices[0]), e_retain_now.to(devices[0])) 
        # loss = eta*loss1 + loss3
        loss = loss3
        # update weights to erase the concept
        loss.backward()
        # wandb.log({"loss": loss.item()})
        pbar.set_postfix({"loss": loss.item()})
        optimizer.step()

    model.eval()
    torch.save({"state_dict": model.state_dict()}, output_name)

def save_model(model, output_name, compvis_config_file=None, device='cpu', save_diffusers=False):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    # folder_path = f'results/{name}'
    # os.makedirs(folder_path, exist_ok=True)
    # if num is not None:
    #     path = f'{folder_path}/{name}-epoch_{num}.pt'
    # else:
    #     path = f'{folder_path}/{name}.pt'
    torch.save({"state_dict": model.state_dict()}, output_name)

    # if save_diffusers:
    #     print('Saving Model in Diffusers Format')
    #     savemodelDiffusers(output_name, compvis_config_file, device=device)

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--train_method', help='method of training', type=str, required=True, choices=['xattn','noxattn', 'selfattn', 'full'])
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--eta', help='trade of between unlearning and retaining objectives', type=int, required=False, default=1)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/train_esd.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt')
    parser.add_argument('--previous_ckpt_path', help='previous forgetting ckpt path for stable diffusion v1-4', type=str, required=False, default='../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--output_path', help='output directory to save results', type=str, required=False, default='results/style50')
    parser.add_argument('--theme', type=str, required=True)
    parser.add_argument('--add_prompts', help='option to add additional prompts', action="store_true", required=False,
                        default=False)
    parser.add_argument('--guided_concepts', help='Concepts to guide the erased concepts', type=str,
    default=None)
    args = parser.parse_args()


    guided_concepts = args.guided_concepts
    concepts = [args.theme]
    old_texts = []
    add_prompts = args.add_prompts
    additional_prompts = []
    if args.theme in theme_available:
        additional_prompts.append('image in {concept} Style')
        additional_prompts.append('art by {concept}')
        additional_prompts.append('artwork by {concept}')
        additional_prompts.append('picture by {concept}')
        additional_prompts.append('style of {concept}')
    else:  # args.theme in class_available
        additional_prompts.append('image of {concept}')
        additional_prompts.append('photo of {concept}')
        additional_prompts.append('portrait of {concept}')
        additional_prompts.append('picture of {concept}')
        additional_prompts.append('painting of {concept}')
    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))
        length = 1 + len(additional_prompts)
        concepts_.extend([concept] * length)

    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
        else:
            new_texts = [[con] * length for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts)
    assert len(new_texts) == len(old_texts)

    index_of_theme = theme_available.index(args.theme)

    prev_forget_concepts = theme_available[:index_of_theme]
    prev_forget_texts = []
    for concept in prev_forget_concepts:
        prev_forget_texts.append(f'{concept}')
        for prompt in additional_prompts:
            prev_forget_texts.append(prompt.format(concept=concept))

    retain_texts = [""]

    for theme in theme_available[index_of_theme+1:]:
        if theme == "Seed_Images":
            theme = "Photo"
        for concept in class_available:
            if concept == args.theme:
                continue
            retain_texts.append(f'A {concept} image in {theme} style')

    # wandb.init(project='quick-canvas-machine-unlearning', name=args.theme, config=args)

    # os.makedirs(args.output_dir, exist_ok=True)
    output_name = f'{args.output_path}'
    print(f"Saving the model to {output_name}")
    
    prompt = f'{args.theme.replace("_", " ")} Style'
    print(f"Old_texts: {old_texts}")
    print(f"New_texts: {new_texts}")
    print(f"Prev_forget_texts: {prev_forget_texts}")
    print(f"Retain_texts: {retain_texts}")
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    eta = args.eta
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    previous_ckpt_path = args.previous_ckpt_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    if args.theme==theme_available[0]:
        print("Train the first model.")
        train_first(prompt=prompt, old_texts=old_texts, new_texts=new_texts, prev_forget_texts=prev_forget_texts, retain_texts=retain_texts, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, eta=eta, config_path=config_path, ckpt_path=ckpt_path, previous_ckpt_path=previous_ckpt_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, output_name=output_name)
    else:
        print("Train in the continual form.")
        train_esd(prompt=prompt, old_texts=old_texts, new_texts=new_texts, prev_forget_texts=prev_forget_texts, retain_texts=retain_texts, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, eta=eta, config_path=config_path, ckpt_path=ckpt_path, previous_ckpt_path=previous_ckpt_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, output_name=output_name)
