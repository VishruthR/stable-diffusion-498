## Single image generation

Generate a single image:

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 1 --n_iter 50 --H 256 --W 256

Generate a batch of 3 images:

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 3 --n_iter 1 --H 256 --W 256

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 10 --n_iter 1

Generate an image from another image:

python scripts/img2img.py --prompt "A cat with a hat" --init-img /users/vishraj/stable-diffusion/outputs/prompt-comparison/grid-0002.png --strength 0.8 --ddim_steps 100

## Measurements

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

### Batch Generations

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
- Use a better GPU with more memory to avoid OOM
- Write better script to generate batches
- Measure memory usage, latency of each image, etc.
- Time to first image, time per image are all relevant
- You need to ensure img2img generations have fewer sampling steps (or maybe not idk)