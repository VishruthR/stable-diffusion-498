## Single image generation

Generate a single image:

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 1 --n_iter 50 --H 256 --W 256

Generate a batch of 3 images:

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 3 --n_iter 1 --H 256 --W 256

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 10 --n_iter 1

Generate an image from another image:

python scripts/img2img.py --prompt "A cat with a hat" --init-img /users/vishraj/stable-diffusion/outputs/prompt-comparison/grid-0002.png --strength 0.8 --ddim_steps 100

## Measurements

### Ampere A30 GPU 24GB (also using better checkpoint)

#### Regular Generation

9 images @ 256x256: inconsistent image quality
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 9 --n_iter 1 --H 256 --W 256
- Samples took 7.97 seconds
- Peak GPU memory allocated: 7757.78 MB

9 images @ 512x512:
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 9 --n_iter 1 --H 512 --W 512
- OOM

### Batch Generation

9 images @ 256x256 with max batch size of 3: Almost as fast w/out batch, but way less memory
- Command: python scripts/batch_gen.py --prompt "a photograph of a cat" --n_samples 9 --max_batch_size 3 --W 256 --H 256
- Samples took 9.36 seconds
- Peak GPU memory allocated: 6413.11 MB

9 images @ 512x512 with max batch size of 3: Does't OOM
- Command: python scripts/batch_gen.py --prompt "a photograph of a cat" --n_samples 9 --max_batch_size 3 --W 512 --H 512
- Samples took 38.01 seconds
- Peak GPU memory allocated: 12118.38 MB
- TTFI: 15.37, TBB: 7.27

512x512 image gen tends to be of much higher quality

Strength set to 0.5 (half as many denoising steps)

9 images @ 256x256 with max batch size of 3: Beat regular batch gen, less memory (image quality not great)
- Command: python scripts/batch_gen.py --prompt "a photograph of a cat" --n_samples 9 --max_batch_size 3 --W 256 --H 256 --strength 0.5
- Samples took 7.86 seconds
- Peak GPU memory allocated: 6413.11 MB

9 images @ 512x512 with max batch size of 3: Faster than strength=0.75, but less variance between images, almost a proportional decrease in TBB
- Command: python scripts/batch_gen.py --prompt "a photograph of a cat" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 0.5
- Samples took 31.24 seconds
- Peak GPU memory allocated: 12118.38 MB
- TTFI: 15.34, TBB: 5.07

Prompt engineering

9 images @ 512x512 with max batch size of 3
- Command: python scripts/batch_gen.py --prompt "a hyperrealistic photograph of a cat with random eye color" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 0.5 --ddim_steps 100
- Samples took 58.72 seconds
- Peak GPU memory allocated: 12118.38 MB
- TTFI: 29.08, TBB: 9.62

Varying total step #

9 images @ 512x512 with max batch size of 3
- Command: python scripts/batch_gen.py --prompt "a hyperrealistic photograph of a cat with random eye color" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 0.5 --ddim_steps 150
- Samples took 91.51 seconds
- Peak GPU memory allocated: 12118.76 MB
- TTFI: 47.69, TBB: 14.32

9 images @ 512x512 with max batch size of 3
- Command: python scripts/batch_gen.py --prompt "a hyperrealistic photograph of a cat" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 0.5 --ddim_steps 200
- Samples took 114.54 seconds
- Peak GPU memory allocated: 12118.76 MB
- TTFI: 56.79, TBB: 18.96

Adjust strength for step size 200

9 images @ 512x512 with max batch size of 3, strength 0.75
- Command: python scripts/batch_gen.py --prompt "a hyperrealistic photograph of a cat" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 0.75 --ddim_steps 200
- Samples took 142.26 seconds
- Peak GPU memory allocated: 12118.76 MB
- TTFI: 56.65, TBB: 28.26

9 images @ 512x512 with max batch size of 3, strength 0.9
- Command: python scripts/batch_gen.py --prompt "a hyperrealistic photograph of a cat" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 0.9 --ddim_steps 200
- Samples took 159.49 seconds
- Peak GPU memory allocated: 12118.76 MB
- TTFI: 56.83, TBB: 33.90

9 images @ 512x512 with max batch size of 3, strength 1.0 (basically image gen from scratch)
- Command: python scripts/batch_gen.py --prompt "a hyperrealistic photograph of a cat" --n_samples 9 --max_batch_size 3 --W 512 --H 512 --strength 1.0 --ddim_steps 200
- Samples took 197.93 seconds
- Peak GPU memory allocated: 12118.76 MB
- TTFI: 66.12, TBB: 43.54

### Old GPU

#### Regular Generation

1 image @ 256x256:
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 1 --n_iter 1 --H 256 --W 256
- Samples took 7.22 seconds
- Peak GPU memory allocated: 6168.76 MB

1 image @ 512x512:
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 1 --n_iter 1 --H 512 --W 512
- Samples took 21.05 seconds
- Peak GPU memory allocated: 7853.29 MB

3 images @ 512x512:
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 3 --n_iter 1 --H 512 --W 512
- OOM

9 images @ 256x256:
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 9 --n_iter 1 --H 256 --W 256
- Samples took 32.77 seconds
- Peak GPU memory allocated: 7757.12 MB

9 images @ 512x512:
- Command: python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 9 --n_iter 1 --H 512 --W 512
- OOM

### Batch Generation

9 images @ 256x256 with max batch size of 3: Almost as fast w/out batch, but way less memory
- Command: python scripts/batch_gen.py --prompt "a photograph of a cat" --n_samples 9 --max_batch_size 3 --W 256 --H 256
- Samples took 36.85 seconds
- Peak GPU memory allocated: 6413.95 MB

3 images @ 512x512 with max batch size of 1: Can now generate 3 images while staying under memory limits. Comparable meory to 1 image gen, faster than generating 3 images back to back
- Command: python scripts/batch_gen.py --prompt "a photograph of a cat" --n_samples 3 --max_batch_size 1 --W 512 --H 512
- Samples took 50.24 seconds
- Peak GPU memory allocated: 7858.01 MB

## Potential benefits of batching image gen

Less memory since you are working with smaller matrices
- Batch size of 10 on 512x512 resolution OOMs
- Batch size of 10 on 256x256 resolution works

Less computation, since you are working with smaller matrices

### Next steps

- Use a better stable diffusion model, this one sucks
- Write better script to generate batches
- Look into ways to speed up the image gen (skipping sampling steps)
- Improve quality of cached image gen
