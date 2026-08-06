# Third-Party Notices

This project is licensed under GNU Affero General Public License version 3
only (AGPL-3.0-only); see [LICENSE](../LICENSE). That project license does not replace or
override the licenses, notices, model terms, trademarks, or data-use
restrictions of third-party components.

This notice identifies principal direct/runtime components and model assets
known to this repository as of 2026-08-06. It is not exhaustive: Python
dependency resolution can install transitive, bundled, or platform-specific
components with additional notices. Distributors must inspect the exact
environment, versions, wheel metadata, and model assets they ship and reproduce
all notices those components require. This file is informational and is not
legal advice.

## Ultralytics YOLO11

The Ultralytics framework and the bundled `yolo11n.pt` checkpoint are
associated with the Ultralytics project.

- Open-source license: GNU Affero General Public License v3.0 (AGPL-3.0).
- Alternative terms: Ultralytics separately offers an Enterprise license. It
  applies only when the user has obtained an applicable license directly from
  Ultralytics. Nothing in this repository grants Enterprise rights.
- Source: <https://github.com/ultralytics/ultralytics>
- Model assets: <https://github.com/ultralytics/assets>
- Licensing information: <https://www.ultralytics.com/license>

Checkpoint use can also be affected by the checkpoint's provenance, training
data, and any metadata supplied with the particular release. Verify those
rights for the intended deployment.

## DeepFace

DeepFace library code is distributed under the MIT License.

- Source and license: <https://github.com/serengil/deepface>

DeepFace wraps separately developed recognition, detection, and demographic
models. DeepFace's code license does not automatically license those models,
their weights, their training data, or their outputs. The upstream project
notes that wrapped models inherit their own license types. Review the selected
backend and downloaded weight files before production, redistribution, or
commercial use.

## MediaPipe

MediaPipe is distributed under the Apache License 2.0.

- Source and license: <https://github.com/google-ai-edge/mediapipe>

## OpenCV and opencv-python

OpenCV's license depends on the release: OpenCV 4.5.0 and later use the Apache
License 2.0, while earlier releases were distributed under the 3-clause BSD
License. The opencv-python project provides unofficial pre-built Python wheels
and has its own packaging and bundled-component notices. Consult the exact
OpenCV version, wheel metadata, and included third-party license files used in
your distribution.

- OpenCV source and license: <https://github.com/opencv/opencv>
- Python wheel packaging: <https://github.com/opencv/opencv-python>

### OpenCV Haar cascade notice

The required OpenCV installation supplies `haarcascade_frontalface_default.xml`.
Its XML header contains the following notice, which must remain with
redistributions of that file:

> Intel License Agreement
> For Open Source Computer Vision Library
>
> Copyright (C) 2000, Intel Corporation, all rights reserved.
> Third party copyrights are property of their respective owners.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistribution's of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
> * Redistribution's in binary form must reproduce the above copyright
>   notice, this list of conditions and the following disclaimer in the
>   documentation and/or other materials provided with the distribution.
> * The name of Intel Corporation may not be used to endorse or promote
>   products derived from this software without specific prior written
>   permission.
>
> This software is provided by the copyright holders and contributors "as is"
> and any express or implied warranties, including, but not limited to, the
> implied warranties of merchantability and fitness for a particular purpose
> are disclaimed. In no event shall the Intel Corporation or contributors be
> liable for any direct, indirect, incidental, special, exemplary, or
> consequential damages (including, but not limited to, procurement of
> substitute goods or services; loss of use, data, or profits; or business
> interruption) however caused and on any theory of liability, whether in
> contract, strict liability, or tort (including negligence or otherwise)
> arising in any way out of the use of this software, even if advised of the
> possibility of such damage.

## NumPy

NumPy is distributed under the 3-clause BSD License. Binary distributions can
bundle additional libraries whose notices are listed in NumPy's own
third-party license files.

- Source and license: <https://github.com/numpy/numpy>

All product names and trademarks belong to their respective owners. Their use
here is for identification only and does not imply endorsement.
