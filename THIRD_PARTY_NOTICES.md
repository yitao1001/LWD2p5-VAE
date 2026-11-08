# Third-party notices

The primary training path directly uses the CompVis computational modules in
`lwd_vae/vendor/compvis/`, pinned to commit
`a506df5756472e2ebaf9078affdde2c4f1502cd4`. Only absolute import prefixes are
changed. Per-file upstream/local hashes are in `provenance.json` and the
upstream license is in `LICENSE`. The unchanged AutoencoderKL source is kept
in `references/compvis/autoencoder_original.py` for audit.

`lwd_vae/finetuning/model.py` wraps these modules in a PyTorch module, adds LWD
shape checks and FP32 posterior statistics under AMP, and replaces the
Lightning lifecycle with an explicit training loop. LPIPS v0.1 calibration
comes from the separately installed `lpips==0.1.4` package; AlexNet comes from
torchvision with explicit user-supplied ImageNet weights. Their respective
upstream licenses and weight terms apply. No pretrained weights are bundled.

The following describes the retained historical baseline:

`lwd_vae/model.py` adapts the KL autoencoder topology, layer naming, residual
blocks, spatial attention, sampling operations, and Gaussian posterior from
[CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion), specifically:

- `ldm/modules/diffusionmodules/model.py`
- `ldm/modules/distributions/distributions.py`
- `ldm/models/autoencoder.py`

The adaptation removes Lightning and unrelated components; adds strict
architecture/shape checks and checkpoint conversion; uses one-channel input
and output, rectangular images, and a configurable model width. No pretrained
weights are distributed here. The upstream code license is reproduced below.

## CompVis latent-diffusion — MIT License

Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
