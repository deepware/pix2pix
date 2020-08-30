### Pix2Pix in Pytorch

Minimalist Pix2Pix implementation in pytorch. 
Paper: https://arxiv.org/abs/1611.07004

The aim of this implementation is easy experimenting.

I followed the paper where possible and filled the blanks with the [official implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). However, I'm not certain that this code is compatible with the paper 100%. Fixes and suggestions are welcome.

#### Usage

1. Download [cityscapes dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz) and extract it. 
2. Run the script with `python3 pix2pix.py cityscapes`

That's it.

#### Notes

No augmentations implemented. Default direction is `B to A` because that's how cityscapes dataset is. You can change it in `load_image` function.

Here's an example output after 50 epochs. The output images are always from validation set.

![sample](https://i.imgur.com/XIGyOB5.png)
