# Diffusar 🐳 - image resoration

Image restoration using diffusions!

The plan is to use customized [Palette](https://arxiv.org/pdf/2111.05826.pdf) model trained on a heavily augmented with *artifacts* open-source datasets to reconstuct images, remove artifacts, noise etc.  
This is still a **WIP**!!!

## To do list

Somewhat sorted by a significance level

- [x] Rewrite [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) in more customizable, readable and human-understandable way
- [x] Write a draft of inference script
- [ ] Flesh out model training and inference
- [ ] Translate model to [PyTorch Lightning ⚡](https://lightning.ai/docs/pytorch/stable/)
- [ ] Add more artifact types!
  - [ ] Heavy corruptions
  - [ ] More noise types!
  - [ ] Cropping *(inpainting / outpainting)*
  - [ ] GAN corruptions *(as in [DiffGAR](https://arxiv.org/pdf/2210.08573.pdf) paper)*
- [ ] Add more datasets *(ideas for datasets)*
  - [x] [COCO](https://paperswithcode.com/dataset/coco) (Precisely a captioning 2017 subset)
  - [ ] [ImageNet](https://paperswithcode.com/dataset/imagenet) *(Might be too big though, but is still a good all round dataset )*
  - [ ] [FFHQ](https://paperswithcode.com/dataset/ffhq) *(good in a moderate amount for training on human images)*
  - [ ] [AFHQ](https://paperswithcode.com/dataset/afhq) *(Good for more pictures of animals)*
  - [ ] [Places](https://paperswithcode.com/dataset/places) *(Good for more PLACES)*
- [ ] Train model
- [ ] Inference with [Gradio 🦄](https://gradio.app/)
- [ ] Write out a comprehensive repository
- [ ] Prepare model deployment
- [ ] Adapt inference pipeline for [Diffusers 🧨](https://huggingface.co/docs/diffusers/index)
- [ ] Control artifacts used in training with config

Also a more of long-shot ideas

- Diverse selection of UNets

## How to use

**TBA**!

## Architecture

More details are **TBA**!

## Related resources

[Unofficial Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
[Paper](https://arxiv.org/pdf/2111.05826.pdf)
[Diffusers 🧨 on instuct-pix2pix](https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix)
[Training of instruct-pix2pix](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py)
[annotated diffusion blog-post](https://huggingface.co/blog/annotated-diffusion)
[denoising-diffusion repository](https://github.com/lucidrains/denoising-diffusion-pytorch)
