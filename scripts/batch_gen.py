"""Batch generation with iterative img2img refinement.

Generates images in batches, using txt2img for initial generation and img2img
for iterative refinement when batch_size > max_batch_size.
"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/batch-gen-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="maximum number of images to generate at once. If n_samples > max_batch_size, will generate iteratively using img2img",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    
    # Set max_batch_size to n_samples if not specified
    if opt.max_batch_size is None:
        opt.max_batch_size = opt.n_samples
    
    # seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    max_batch_size = opt.max_batch_size
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        prompts_list = [prompt]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            prompts_list = f.read().splitlines()

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Setup for img2img if we need iterative generation
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    
    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    
    if batch_size > max_batch_size:
        print(f"Will generate {batch_size} images in multiple batches of up to {max_batch_size}")
        print(f"Using img2img with strength {opt.strength} (t_enc={t_enc}) for iterative generation")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                torch.cuda.reset_peak_memory_stats()
                tic = time.time()
                all_samples = list()
                
                for prompt in prompts_list:
                    for n in trange(opt.n_iter, desc="Sampling"):
                        # Initialize current samples list for this iteration
                        current_samples = []
                        images_generated = 0
                        generation_round = 0
                        
                        # Generate images in batches until we reach n_samples
                        while images_generated < batch_size:
                            generation_round += 1
                            # Determine how many images to generate in this batch
                            current_batch_size = min(max_batch_size, batch_size - images_generated)
                            
                            round_start = time.time()
                            
                            if generation_round == 1:
                                # Round 1: Generate initial images from scratch using txt2img
                                print(f"Round {generation_round}: Generating {current_batch_size} images from scratch (txt2img)")
                                
                                uc = None
                                if opt.scale != 1.0:
                                    uc = model.get_learned_conditioning(current_batch_size * [""])
                                
                                c = model.get_learned_conditioning(current_batch_size * [prompt])
                                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                
                                samples, _ = sampler.sample(
                                    S=opt.ddim_steps,
                                    conditioning=c,
                                    batch_size=current_batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta
                                )
                                
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                
                            else:
                                # Round 2+: Generate new images using img2img from previous batch
                                # Each image from the previous batch serves as input for one new image
                                print(f"Round {generation_round}: Generating {current_batch_size} images from previous batch (img2img)")
                                
                                # Use the first current_batch_size images from previous round as input
                                source_images = current_samples[:current_batch_size]
                                source_images_stacked = torch.stack(source_images, dim=0)
                                
                                # Encode to latent space (convert from [0,1] to [-1,1] range)
                                init_latent = model.get_first_stage_encoding(
                                    model.encode_first_stage(source_images_stacked * 2.0 - 1.0)
                                )
                                
                                uc = None
                                if opt.scale != 1.0:
                                    uc = model.get_learned_conditioning(current_batch_size * [""])
                                
                                c = model.get_learned_conditioning(current_batch_size * [prompt])
                                
                                # Add noise to latent (controlled by strength parameter)
                                z_enc = sampler.stochastic_encode(
                                    init_latent, 
                                    torch.tensor([t_enc] * current_batch_size).to(device)
                                )
                                
                                # Denoise to generate new image
                                samples = sampler.decode(
                                    z_enc, c, t_enc,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc
                                )
                                
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            
                            # Save individual samples
                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1
                            
                            # Add to current samples for potential next round
                            for i in range(x_samples.shape[0]):
                                current_samples.append(x_samples[i])
                            
                            images_generated += current_batch_size
                            round_end = time.time()
                            print(f"Generated {images_generated}/{batch_size} images (Round {generation_round} took {round_end - round_start:.2f} seconds)")
                        
                        # Add all samples from this iteration to all_samples
                        all_samples.extend(current_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n c h w -> n c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    
    print(f"\nSamples took {toc - tic:.2f} seconds")
    
    current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Current GPU memory allocated: {current_mem:.2f} MB")
    print(f"Peak GPU memory allocated: {peak_memory:.2f} MB")


if __name__ == "__main__":
    main()
