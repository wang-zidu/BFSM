# Third-party notices

The BFSM Research License applies to BFSM's own contributions. The components below retain their notices and original licenses. Model/data terms are separate from source-code licenses.

| Included source portions | Origin | Notice |
| --- | --- | --- |
| Nonrigid registration | [wuhaozhe/pytorch-nicp](https://github.com/wuhaozhe/pytorch-nicp) | [Full notice](#pytorch-nicp) |
| Reconstruction network/preprocessing | [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) | [Full notice](#deep3dfacereconpytorch) |
| ResNet portions | [torchvision](https://github.com/pytorch/vision) | [Full notice](#torchvision) |
| Detection, landmarks, renderer and OBJ export portions | [3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3/tree/e15385837dc1e051a6cf376b3827f2e279537b29) | [Full notice](#3ddfa-v3) |
| Detector portions | [ternaus/retinaface](https://github.com/ternaus/retinaface/tree/04bf71613a4b03676abe91216d55f9cad68a183b) | [Full notice](#retinaface) |
| Deterministic raster derivative adapter | [PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba) | [Full notice](#pytorch3d) |
| Embedded offline viewer | [Plotly.js](https://github.com/plotly/plotly.js) | `modules/02_morphable_models/src/bfsm_models/PLOTLY_LICENSE.txt`; notice also accompanies generated HTML |

## Registration and geometry implementation

The registration interface and numerical conventions acknowledge [3DMM-Fitting-Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch). Its model, loss and utility interfaces are reimplemented here using PCA linear decoding, perspective projection, spherical-harmonic lighting and standard tensor losses. This is not a relicensing of upstream source. Numerical compatibility is documented in the registration guide.

The renderer's barycentric calculation and detector box/landmark decoding use BFSM implementations of the underlying mathematical operations. Existing notices for the remaining renderer and detector source are retained. No original FLAME converter implementation or DECA source is vendored.

## Externally obtained models

- **BFM09 and expression data:** obtain and accept the [Basel model terms](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) and corresponding expression-model terms. Original data and converted BFM-derived base models are obtained/prepared externally, not redistributed by this repository.
- **Image checkpoints:** original-provider download links and local filenames are listed in [assets/README.md](assets/README.md). The closed-eye checkpoint is Deep3DFaceRecon's original release. A source-code license does not replace model/source-data terms.
- **FLAME2020, masks and correspondence:** obtain under [FLAME model terms](https://flame.is.tue.mpg.de/modellicense.html) and the [converter's applicable notice](https://github.com/TimoBolkart/BFM_to_FLAME). Original models, topology/correspondence inputs and FLAME-derived geometry are external/local assets. Provider access is not permission to redistribute them.
- **DECA:** obtain encoder source and weights through [DECA](https://github.com/YadiraF/DECA) under its terms. The BFSM frontend uses the external encoder with FLAME expression/pose decoding and camera projection. DECA retains its own terms.

## Example images

[Image sources](data/input/IMAGE_SOURCES.md).

<a id="3ddfa-v3"></a>

## 3DDFA-V3

```text
MIT License

Copyright (c) 2024 Zidu Wang

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
```

<a id="deep3dfacereconpytorch"></a>

## Deep3DFaceRecon_pytorch

```text
MIT License

Copyright (c) 2022 Sicheng Xu

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
```

<a id="pytorch3d"></a>

## PyTorch3D

```text
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

<a id="retinaface"></a>

## RetinaFace

```text
MIT License

Copyright (c) 2020 Vladimir Iglovikov

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
```

<a id="pytorch-nicp"></a>

## pytorch-nicp

```text
MIT License

Copyright (c) 2021 Haozhe Wu

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
```

<a id="torchvision"></a>

## torchvision

```text
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
