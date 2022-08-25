# Update v0.6: Added support for weighted prompts (based on the code from @lstein's [repo](https://github.com/lstein/stable-diffusion))

- You can now use weighted prompts to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.
- The number followed by the colon represents the weight given to the words before the colon.
  The weights can be fractions or integers.

# Optimized Stable Diffusion (Sort of)

This repo is a modified version of the Stable Diffusion repo, optimized to use lesser VRAM than the original by sacrificing on inference speed.

img2img to generate new image based on a prior image and prompt

- `optimized_img2img.py` Generate images using CLI
- `img2img_gradio.py` Generate images using gradio GUI

txt2img to generate an image based only on a prompt

- `optimized_txt2img.py` Generate images using CLI
- `txt2img_gradio.py` Generate images using gradio GUI

### img2img

- It can generate _512x512 images from a prior image and prompt on a 4GB VRAM GPU in under 20 seconds per image_ (RTX 2060 in my case).

- You can use the `--H` & `--W` arguments to set the size of the generated image.The maximum size that can fit on 6GB GPU (RTX 2060) is around 576x768.

- For example, the following command will generate 20 512x512 images:

`python optimizedSD/optimized_img2img.py --prompt "Austrian alps" --init-img ~/sketch-mountains-input.jpg --strength 0.8 --n_iter 2 --n_samples 5 --H 576 --W 768`

### txt2img

- It can generate _512x512 images from a prompt on a 4GB VRAM GPU in under 25 seconds per image_ (RTX 2060 in my case).

- You can use the `--H` & `--W` arguments to set the size of the generated image.

- For example, the following command will generate 20 512x512 images:

`python optimizedSD/optimized_txt2img.py --prompt "Cyberpunk style image of a Telsa car reflection in rain" --H 512 --W 512 --seed 27 --n_iter 2 --n_samples 10 --ddim_steps 50`

---

### `--seed` (Seed)

- The code will give the seed number along with each generated image. To generate the same image again, just specify the seed using `--seed` argument. Also, images will be saved with its seed number as its name.

- eg. If the seed number for an image is `1234` and it's the 55th image in the folder, the image name will be named `seed_1234_00055.png`. If no seed is given as an argument, a random initial seed will be choosen.

### `--n_samples` (batch size)

- To get the lowest inference time per image, use the maximum batch size `--n_samples` that can fit on the GPU. Inference time per image will reduce on increasing the batch size, but the required VRAM will also increase.

- If you get a CUDA out of memory error, try reducing the batch size `--n_samples`. If it doesn't work, the other option is to reduce the image width `--W` or height `--H` or both.

### `--H` & `--W` (Height & Width)

- Both height and width should be a multiple of 64

### `--precision autocast` or `--precision full` (Full or Mixed Precision)

- Mixed Precision is enabled by default. If you don't have a GPU with tensor cores, you can still use mixed precision to run the code using lesser VRAM but the inference time may be larger. And if you feel that the inference is slower, try using the `--precision full` argument to disable it.

### Gradio for Graphical User Interface

- You can also use gradio interface for img2img & txt2img instead of the CLI. Just activate the conda env and install the latest version of gradio using `pip install gradio` .

- Run img2img using `python optimizedSD/img2img_gradio` and txt2img using `python optimizedSD/img2img_gradio`.

### Weighted Prompts

- The prompts can also be weighted to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.

- The number followed by the colon represents the weight given to the words before the colon.The weights can be both fractions or integers.

### Installation

- All the modified files are in the [optimizedSD](optimizedSD) folder, so if you have already cloned the original repo, you can just download and copy this folder into the orignal repo instead of cloning the entire repo. You can also clone this repo and follow the same installation steps as the original repo(mainly creating the conda env and placing the weights at the specified location).

---

- To achieve this, the stable diffusion model is fragmented into four parts which are sent to the GPU only when needed. After the calculation is done, they are moved back to the CPU. This allows us to run a bigger model on a lower VRAM.

- The only drawback is higher inference time which is still an order of magnitude faster than inference on CPU.

## Changelog

- v0.6: Added support for using weighted prompts. (based on @lstein [repo](https://github.com/lstein/stable-diffusion))
- v0.5: Added support for using gradio interface.
- v0.4: Added support for specifying image seed.
- v0.3: Added support for using mixed precision.
- v0.2: Added support for generating images in batches.
- v0.1: Split the model into multiple parts to run it on lower VRAM.

_Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:_

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1) is a latent text-to-image diffusion
model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), we were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database.
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487),
this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See [this section](#stable-diffusion-v1) below and the [model card](https://huggingface.co/CompVis/stable-diffusion).

## Requirements

A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

## Stable Diffusion v1

Stable Diffusion v1 refers to a specific configuration of the model
architecture that uses a downsampling-factor 8 autoencoder with an 860M UNet
and CLIP ViT-L/14 text encoder for the diffusion model. The model was pretrained on 256x256 images and
then finetuned on 512x512 images.

*Note: Stable Diffusion v1 is a general text-to-image diffusion model and therefore mirrors biases and (mis-)conceptions that are present
in its training data. 
Details on the training procedure and data, as well as the intended use of the model can be found in the corresponding [model card](Stable_Diffusion_v1_Model_Card.md).*

The weights are available via [the CompVis organization at Hugging Face](https://huggingface.co/CompVis) under [a license which contains specific use-based restrictions to prevent misuse and harm as informed by the model card, but otherwise remains permissive](LICENSE). While commercial use is permitted under the terms of the license, **we do not recommend using the provided weights for services or products without additional safety mechanisms and considerations**, since there are [known limitations and biases](Stable_Diffusion_v1_Model_Card.md#limitations-and-bias) of the weights, and research on safe and ethical deployment of general text-to-image models is an ongoing effort. **The weights are research artifacts and should be treated as such.**

[The CreativeML OpenRAIL M license](LICENSE) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.

### Weights

We currently provide the following checkpoints:

- `sd-v1-1.ckpt`: 237k steps at resolution `256x256` on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en).
  194k steps at resolution `512x512` on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) (170M examples from LAION-5B with resolution `>= 1024x1024`).
- `sd-v1-2.ckpt`: Resumed from `sd-v1-1.ckpt`.
  515k steps at resolution `512x512` on [laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/) (a subset of laion2B-en with estimated aesthetics score `> 5.0`, and additionally
filtered to images with an original size `>= 512x512`, and an estimated watermark probability `< 0.5`. The watermark estimate is from the [LAION-5B](https://laion.ai/blog/laion-5b/) metadata, the aesthetics score is estimated using the [LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)).
- `sd-v1-3.ckpt`: Resumed from `sd-v1-2.ckpt`. 195k steps at resolution `512x512` on "laion-aesthetics v2 5+" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).
- `sd-v1-4.ckpt`: Resumed from `sd-v1-2.ckpt`. 225k steps at resolution `512x512` on "laion-aesthetics v2 5+" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).

Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,
5.0, 6.0, 7.0, 8.0) and 50 PLMS sampling
steps show the relative improvements of the checkpoints:
![sd evaluation results](assets/v1-variants-scores.jpg)

### Text-to-Image with Stable Diffusion

![txt2img-stable2](assets/stable-samples/txt2img/merged-0005.png)
![txt2img-stable2](assets/stable-samples/txt2img/merged-0007.png)

Stable Diffusion is a latent diffusion model conditioned on the (non-pooled) text embeddings of a CLIP ViT-L/14 text encoder.
We provide a [reference script for sampling](#reference-sampling-script), but
there also exists a [diffusers integration](#diffusers-integration), which we
expect to see more active community development.

#### Reference Sampling Script

We provide a reference sampling script, which incorporates

- a [Safety Checker Module](https://github.com/CompVis/stable-diffusion/pull/36),
  to reduce the probability of explicit outputs,
- an [invisible watermarking](https://github.com/ShieldMnt/invisible-watermark)
  of the outputs, to help viewers [identify the images as machine-generated](scripts/tests/test_watermark.py).

After [obtaining the `stable-diffusion-v1-*-original` weights](#weights), link them
```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
```

and sample with

```
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms
```

By default, this uses a guidance scale of `--scale 7.5`, [Katherine Crowson's implementation](https://github.com/CompVis/latent-diffusion/pull/51) of the [PLMS](https://arxiv.org/abs/2202.09778) sampler,
and renders images of size 512x512 (which it was trained on) in 50 steps. All supported arguments are listed below (type `python scripts/txt2img.py --help`).


```commandline
usage: txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--skip_grid] [--skip_save] [--ddim_steps DDIM_STEPS] [--plms] [--laion400m] [--fixed_code] [--ddim_eta DDIM_ETA]
                  [--n_iter N_ITER] [--H H] [--W W] [--C C] [--f F] [--n_samples N_SAMPLES] [--n_rows N_ROWS] [--scale SCALE] [--from-file FROM_FILE] [--config CONFIG] [--ckpt CKPT]
                  [--seed SEED] [--precision {full,autocast}]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     the prompt to render
  --outdir [OUTDIR]     dir to write results to
  --skip_grid           do not save a grid, only individual samples. Helpful when evaluating lots of samples
  --skip_save           do not save individual samples. For speed measurements.
  --ddim_steps DDIM_STEPS
                        number of ddim sampling steps
  --plms                use plms sampling
  --laion400m           uses the LAION400M model
  --fixed_code          if enabled, uses the same starting code across samples
  --ddim_eta DDIM_ETA   ddim eta (eta=0.0 corresponds to deterministic sampling
  --n_iter N_ITER       sample this often
  --H H                 image height, in pixel space
  --W W                 image width, in pixel space
  --C C                 latent channels
  --f F                 downsampling factor
  --n_samples N_SAMPLES
                        how many samples to produce for each given prompt. A.k.a. batch size
  --n_rows N_ROWS       rows in the grid (default: n_samples)
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --from-file FROM_FILE
                        if specified, load prompts from this file
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of model
  --seed SEED           the seed (for reproducible sampling)
  --precision {full,autocast}
                        evaluate at this precision
```

Note: The inference config for all v1 versions is designed to be used with EMA-only checkpoints.
For this reason `use_ema=False` is set in the configuration, otherwise the code will try to switch from
non-EMA to EMA weights. If you want to examine the effect of EMA vs no EMA, we provide "full" checkpoints
which contain both types of weights. For these, `use_ema=False` will load and use the non-EMA weights.


#### Diffusers Integration

A simple way to download and sample Stable Diffusion is by using the [diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers):
```py
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4",
	use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
```


### Image Modification with Stable Diffusion

By using a diffusion-denoising mechanism as first proposed by [SDEdit](https://arxiv.org/abs/2108.01073), the model can be used for different
tasks such as text-guided image-to-image translation and upscaling. Similar to the txt2img sampling script,
we provide a script to perform image modification with Stable Diffusion.

The following describes an example where a rough sketch made in [Pinta](https://www.pinta-project.com/) is converted into a detailed artwork.

```
python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8
```

Here, strength is a value between 0.0 and 1.0, that controls the amount of noise that is added to the input image.
Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input. See the following example.

**Input**

![sketch-in](assets/stable-samples/img2img/sketch-mountains-input.jpg)

**Outputs**

![out3](assets/stable-samples/img2img/mountains-3.png)
![out2](assets/stable-samples/img2img/mountains-2.png)

This procedure can, for example, also be used to upscale samples from the base model.

## Comments

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
  and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
  Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories).

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
