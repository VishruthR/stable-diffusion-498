## Single image generation

Generate a single image:

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 1 --n_iter 50 --H 256 --W 256

Generate a batch of 3 images:

python scripts/txt2img.py --prompt "a photograph of a cat" --plms --n_samples 3 --n_iter 1 --H 256 --W 256