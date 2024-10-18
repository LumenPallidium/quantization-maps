## Introduction

This is a repo for vector quantization and [modern self-organizing maps (SOMs)](https://arxiv.org/abs/2302.07950). Additionally, I added a differential version of these SOMs based on [this](https://arxiv.org/abs/1806.02199). I also include a test on CIFAR which makes pretty pictures, visualized below!


https://github.com/LumenPallidium/audio_generation/assets/42820488/567d3d5b-27e1-4cbb-b5f1-44c1cc9536c1

Here is a similar visualization of the SOM codebook usage when reconstructing audio (which was admittedly not as cool as I hoped it would be):

https://github.com/LumenPallidium/audio_generation/assets/42820488/a509110c-b194-4358-9888-eb99f1c88c3a

Here, each color represents a different codebook, so the plot is displaying the entries producing the sound at that instant in the video. I was hoping it would not look so random. Nonetheless, I hope that SOMs will serve to make attention mechanisms on the codebook more robust (since the codebook has a natural notion of neighborhood and proximity among entries), as well as giving structure to "interpolation" between codebook entries.


## Installation

To install this package, run:

```
pip install git+https://github.com/LumenPallidium/quantization-maps.git
```

The only requirements are Pytorch (>=2.0) and einops. The above PIP install command will install Pytorch, but I would reccomend installing on your own independently, so you can configure any neccesary environments, CUDA tools, etc.




