### 输入

```
<p align="center">
    <!-- community badges -->
    <a href="https://discord.gg/uMbNqcraFc"><img src="https://dcbadge.vercel.app/api/server/uMbNqcraFc?style=plastic"/></a>
    <!-- doc badges -->
    <a href='https://docs.nerf.studio/'>
        <img src='https://readthedocs.com/projects/plenoptix-nerfstudio/badge/?version=latest' alt='Documentation Status' /></a>
    <!-- pi package badge -->
    <a href="https://badge.fury.io/py/nerfstudio"><img src="https://badge.fury.io/py/nerfstudio.svg" alt="PyPI version"></a>
    <!-- code check badges -->
    <a href='https://github.com/nerfstudio-project/nerfstudio/actions/workflows/core_code_checks.yml'>
        <img src='https://github.com/nerfstudio-project/nerfstudio/actions/workflows/core_code_checks.yml/badge.svg' alt='Test Status' /></a>
    <!-- license badge -->
    <a href="https://github.com/nerfstudio-project/nerfstudio/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/_images/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nerf.studio/_images/logo.png">
    <!-- /pypi-strip -->
    <img alt="nerfstudio" src="https://docs.nerf.studio/_images/logo.png" width="400">
    <!-- pypi-strip -->
    </picture>
    <!-- /pypi-strip -->
</p>

<!-- Use this for pypi package (and disable above). Hacky workaround -->
<!-- <p align="center">
    <img alt="nerfstudio" src="https://docs.nerf.studio/_images/logo.png" width="400">
</p> -->

<p align="center"> A collaboration friendly studio for NeRFs </p>

<p align="center">
    <a href="https://docs.nerf.studio">
        <img alt="documentation" src="https://user-images.githubusercontent.com/3310961/194022638-b591ce16-76e3-4ba6-9d70-3be252b36084.png" width="150"></a>
    <a href="https://viewer.nerf.studio/">
        <img alt="viewer" src="https://user-images.githubusercontent.com/3310961/194022636-a9efb85a-14fd-4002-8ed4-4ca434898b5a.png" width="150"></a>
    <a href="https://colab.research.google.com/github/nerfstudio-project/nerfstudio/blob/main/colab/demo.ipynb">
        <img alt="colab" src="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/docs/_static/imgs/readme_colab.png" width="150"></a>
</p>

<img src="https://user-images.githubusercontent.com/3310961/194017985-ade69503-9d68-46a2-b518-2db1a012f090.gif" width="52%"/> <img src="https://user-images.githubusercontent.com/3310961/194020648-7e5f380c-15ca-461d-8c1c-20beb586defe.gif" width="46%"/>

- [快速开始](#快速开始)
- [了解更多](#了解更多)
- [支持的特性](#支持的特性)

# 关于

_使用 nerfstudio，一切就像即插即用一样简单！_

Nerfstudio 提供了一个简单的 API，简化了创建、训练和测试神经辐射场（NeRFs）的端到端流程。
该库支持通过模块化每个组件，对神经辐射场进行更具可解释性的实现。
通过更模块化的神经辐射场，我们希望为用户提供一个更友好的技术探索体验。

这是一个对贡献者友好的仓库，目标是建立一个社区，让用户可以更轻松地在彼此的贡献基础上进行构建。
Nerfstudio 最初是由加州大学伯克利分校 [KAIR 实验室](https://people.eecs.berkeley.edu/~kanazawa/index.html#kair) 的学生在 [伯克利人工智能研究中心（BAIR）](https://bair.berkeley.edu/) 于 2022 年 10 月作为一个研究项目开源发布的（[论文](https://arxiv.org/abs/2302.04264)）。目前由伯克利的学生和社区贡献者共同开发。

我们致力于提供学习资源，帮助你了解神经辐射场的基础知识（如果你是初学者），并与最新技术保持同步（如果你是经验丰富的老手）。作为研究人员，我们深知掌握这项下一代技术有多难。所以我们通过教程、文档等方式来提供帮助！

有功能请求吗？想添加你全新的神经辐射场模型吗？有新的数据集吗？**我们欢迎[贡献](https://docs.nerf.studio/reference/contributing.html)！** 如果你有任何问题，请随时通过 [Discord](https://discord.gg/uMbNqcraFc) 联系 nerfstudio 团队。

有反馈吗？如果你想告诉我们你是谁、为什么对 Nerfstudio 感兴趣或提供任何反馈，我们希望你能填写我们的 [Nerfstudio 反馈表](https://forms.gle/sqN5phJN7LfQVwnP9)！

我们希望 nerfstudio 能让你更快地构建 :hammer: 共同学习 :books: 并为我们的神经辐射场社区做出贡献 :sparkling_heart:。

## 赞助商

这项工作的赞助商包括 [Luma AI](https://lumalabs.ai/) 和 [BAIR 共享资源](https://bcommons.berkeley.edu/home)。

<p align="left">
    <a href="https://lumalabs.ai/">
        <!-- pypi-strip -->
        <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/luma_dark.png">
        <source media="(prefers-color-scheme: light)" srcset="docs/_static/imgs/luma_light.png">
        <!-- /pypi-strip -->
        <img alt="Luma AI" src="docs/_static/imgs/luma_light.png" width="300">
        <!-- pypi-strip -->
        </picture>
        <!-- /pypi-strip -->
    </a>
    <a href="https://bcommons.berkeley.edu/home">
        <!-- pypi-strip -->
        <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/bair_dark.png">
        <source media="(prefers-color-scheme: light)" srcset="docs/_static/imgs/bair_light.png">
        <!-- /pypi-strip -->
        <img alt="BAIR" src="docs/_static/imgs/bair_light.png" width="300">
        <!-- pypi-strip -->
        </picture>
        <!-- /pypi-strip -->
    </a>
</p>

# 快速开始

本快速开始指南将帮助你使用经典的 Blender Lego 场景训练默认的香草神经辐射场（vanilla NeRF）。
对于更复杂的更改（例如，使用你自己的数据运行/设置新的神经辐射场图），请参考我们的[参考资料](#了解更多)。

## 1. 安装：设置环境

### 先决条件

你必须拥有安装了 CUDA 的 NVIDIA 显卡。该库已在 CUDA 11.8 版本上进行了测试。你可以在[此处](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)找到更多关于安装 CUDA 的信息。

### 创建环境

Nerfstudio 需要 `python >= 3.8`。我们建议使用 conda 来管理依赖项。在继续之前，请确保安装了 [Conda](https://docs.conda.io/miniconda.html)。

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
pip install --upgrade pip
```

### 依赖项

安装带有 CUDA 的 PyTorch（此仓库已在 CUDA 11.7 和 CUDA 11.8 上进行了测试）和 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)。
`cuda-toolkit` 是构建 `tiny-cuda-nn` 所必需的。

对于 CUDA 11.8：

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

更多信息请参阅安装文档中的[依赖项](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)部分。

### 安装 nerfstudio

简单选项：

```bash
pip install nerfstudio
```

**或者** 如果你想要最新版本：

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

**或者** 如果你想跳过所有安装步骤，直接开始使用 nerfstudio，可以使用 Docker 镜像：

请参阅[安装](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md) - **使用 Docker 镜像**部分。

## 2. 训练你的第一个模型！

以下命令将训练一个 _nerfacto_ 模型，这是我们推荐用于真实场景的模型。

```bash
# 下载一些测试数据：
ns-download-data nerfstudio --capture-name=poster
# 训练模型
ns-train nerfacto --data data/nerfstudio/poster
```

如果一切正常，你应该会看到如下的训练进度：

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766069-cadfd34f-8833-4156-88b7-ad406d688fc0.png">
</p>

在终端末尾的链接上导航将加载网页查看器。如果你在远程机器上运行，则需要转发 WebSocket 端口（默认为 7007）。

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766653-586a0daa-466b-4140-a136-6b02f2ce2c54.png">
</p>

### 从检查点恢复训练 / 可视化现有运行

可以通过以下命令加载预训练模型：

```bash
ns-train nerfacto --data data/nerfstudio/poster --load-dir {outputs/.../nerfstudio_models}
```

## 可视化现有运行

给定一个预训练模型的检查点，你可以通过以下命令启动查看器：

```bash
ns-viewer --load-config {outputs/.../config.yml}
```

## 3. 导出结果

一旦你有了一个神经辐射场模型，你可以渲染出一个视频或导出一个点云。

### 渲染视频

首先，我们必须为相机创建一个路径。这可以在查看器的“渲染”选项卡下完成。将你的 3D 视图定向到你希望视频开始的位置，然后按下“添加相机”。这将设置第一个相机关键帧。继续移动到新的视点，添加更多的相机以创建相机路径。我们提供了其他参数来进一步优化你的相机路径。满意后，按下“渲染”，这将显示一个模态框，其中包含渲染视频所需的命令。终止训练作业（或者如果你有足够的计算资源，可以创建一个新的终端）并运行该命令以生成视频。

还有其他视频导出选项，通过运行以下命令了解更多信息：

```bash
ns-render --help
```

### 生成点云

虽然神经辐射场模型并非专门用于生成点云，但仍然可以实现。在 3D 查看器中导航到“导出”选项卡，选择“点云”。如果选择了裁剪选项，黄色方块内的所有内容将被导出为点云。根据需要修改设置，然后在命令行中运行面板底部的命令。

或者，你可以不使用查看器，直接使用命令行界面。通过运行以下命令了解导出选项：

```bash
ns-export pointcloud --help
```

## 4. 使用自定义数据

使用现有的数据集很不错，但你可能希望使用自己的数据！我们支持多种使用自定义数据的方法。在将数据用于 nerfstudio 之前，必须确定相机的位置和方向，然后使用 `ns-process-data` 将其转换为我们的格式。我们依赖外部工具来完成此操作，相关说明和信息可以在文档中找到。

| 数据                                                                                      | 采集设备                                                                    | 要求                                                                                                                | `ns-process-data` 速度 |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| 📷[图像](https://docs.nerf.studio/quickstart/custom_dataset.html#images-or-video)            | 任意                                                                        | [COLMAP](https://colmap.github.io/install.html)                                                                        | 🐢                       |
| 📹[视频](https://docs.nerf.studio/quickstart/custom_dataset.html#images-or-video)            | 任意                                                                        | [COLMAP](https://colmap.github.io/install.html)                                                                        | 🐢                       |
| 🌎[360 度数据](https://docs.nerf.studio/quickstart/custom_dataset.html#data-equirectangular) | 任意                                                                        | [COLMAP](https://colmap.github.io/install.html)                                                                        | 🐢                       |
| 📱[Polycam](https://docs.nerf.studio/quickstart/custom_dataset.html#polycam-capture)         | 支持 LiDAR 的 iOS 设备                                                      | [Polycam 应用](https://poly.cam/)                                                                                      | 🐇                       |
| 📱[KIRI 引擎](https://docs.nerf.studio/quickstart/custom_dataset.html#kiri-engine-capture)   | iOS 或 Android 设备                                                         | [KIRI 引擎应用](https://www.kiriengine.com/)                                                                           | 🐇                       |
| 📱[Record3D](https://docs.nerf.studio/quickstart/custom_dataset.html#record3d-capture)       | 支持 LiDAR 的 iOS 设备                                                      | [Record3D 应用](https://record3d.app/)                                                                                 | 🐇                       |
| 📱[Spectacular AI](https://docs.nerf.studio/quickstart/custom_dataset.html#spectacularai)    | iOS、OAK 及[其他设备](https://www.spectacularai.com/mapping#supported-devices) | [应用](https://apps.apple.com/us/app/spectacular-rec/id6473188128) / [`sai-cli`](https://www.spectacularai.com/mapping) | 🐇                       |
| 🖥[Metashape](https://docs.nerf.studio/quickstart/custom_dataset.html#metashape)             | 任意                                                                        | [Metashape](https://www.agisoft.com/)                                                                                  | 🐇                       |
| 🖥[RealityCapture](https://docs.nerf.studio/quickstart/custom_dataset.html#realitycapture)   | 任意                                                                        | [RealityCapture](https://www.capturingreality.com/realitycapture)                                                      | 🐇                       |
| 🖥[ODM](https://docs.nerf.studio/quickstart/custom_dataset.html#odm)                         | 任意                                                                        | [ODM](https://github.com/OpenDroneMap/ODM)                                                                             | 🐇                       |
| 👓[Aria](https://docs.nerf.studio/quickstart/custom_dataset.html#aria)                       | Aria 眼镜                                                                   | [Project Aria](https://projectaria.com/)                                                                               | 🐇                       |
| 🛠[自定义](https://docs.nerf.studio/quickstart/data_conventions.html)                        | 任意                                                                        | 相机位姿                                                                                                            | 🐇                       |

## 5. 高级选项

### 训练除 nerfacto 之外的模型

我们提供了除 nerfacto 之外的其他模型，例如，如果你想训练原始的神经辐射场模型，可以使用以下命令：

```bash
ns-train vanilla-nerf --data DATA_PATH
```

要获取包含的所有模型的完整列表，请运行 `ns-train --help`。

### 修改配置

每个模型都包含许多可以更改的参数，这里无法一一列出。使用 `--help` 命令查看完整的配置选项列表。

```bash
ns-train nerfacto --help
```

### Tensorboard / WandB / 查看器

我们支持四种不同的方法来跟踪训练进度，包括使用查看器、[Tensorboard](https://www.tensorflow.org/tensorboard)、[Weights and Biases](https://wandb.ai/site) 和 [Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=github)。你可以通过在训练命令后附加 `--vis {viewer, tensorboard, wandb, comet viewer+wandb, viewer+tensorboard, viewer+comet}` 来指定使用哪个可视化工具。同时使用查看器和 wandb 或 tensorboard 可能会在评估步骤中导致卡顿问题。查看器仅适用于速度较快的方法（例如 nerfacto、instant-ngp），对于像 NeRF 这样较慢的方法，请使用其他日志记录器。

# 了解更多

以上就是使用 nerfstudio 基础知识的入门指南。

如果你想了解更多关于如何创建自己的管道、使用查看器进行开发、运行基准测试等内容，请查看以下快速链接或直接访问我们的[文档](https://docs.nerf.studio/)。

| 部分                                                                    | 描述                                                             |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [文档](https://docs.nerf.studio/)                                          | 完整的 API 文档和教程                                            |
| [查看器](https://viewer.nerf.studio/)                                      | 我们的网页查看器主页                                             |
| 🎒**教育资源**                                                    |                                                                  |
| [模型描述](https://docs.nerf.studio/nerfology/methods/index.html)          | 对 nerfstudio 支持的所有模型的描述以及组件部分的解释。           |
| [组件描述](https://docs.nerf.studio/nerfology/model_components/index.html) | 交互式笔记本，解释各种模型中值得注意/常用的模块。                |
| 🏃**教程**                                                        |                                                                  |
| [入门指南](https://docs.nerf.studio/quickstart/installation.html)          | 一个更深入的指南，介绍如何从安装到贡献开始使用 nerfstudio。      |
| [使用查看器](https://docs.nerf.studio/quickstart/viewer_quickstart.html)   | 一个关于如何导航查看器的快速演示视频。                           |
| [使用 Record3D](https://www.youtube.com/watch?v=XwKq7qDQCQk)               | 一个关于如何在不使用 COLMAP 的情况下运行 nerfstudio 的演示视频。 |
| 💻**开发者资源**                                                  |                                                                  |
| [创建管道](https://docs.nerf.studio/developer_guides/pipelines/index.html) | 学习如何通过使用和/或实现新模块轻松构建新的神经渲染管道。        |
| [创建数据集](https://docs.nerf.studio/quickstart/custom_dataset.html)      | 有新的数据集？学习如何在 nerfstudio 中运行它。                   |
| [贡献指南](https://docs.nerf.studio/reference/contributing.html)           | 关于如何开始贡献的详细步骤。                                     |
| 💖**社区**                                                        |                                                                  |
| [Discord](https://discord.gg/uMbNqcraFc)                                   | 加入我们的社区进行更多讨论。我们很乐意听取你的意见！             |
| [Twitter](https://twitter.com/nerfstudioteam)                              | 在 Twitter 上关注我们 @nerfstudioteam，了解酷炫的更新和公告      |
| [反馈表](TODO)                                                             | 我们欢迎任何反馈！这是我们了解你使用 Nerfstudio 的目的的机会。   |

# 支持的特性

我们提供以下支持结构，以简化神经辐射场的入门学习。

**如果你正在寻找当前不支持的功能，请随时通过 [Discord](https://discord.gg/uMbNqcraFc) 联系 Nerfstudio 团队！**

- 🔎 基于 Web 的可视化工具，允许你：
  - 实时可视化训练过程并与场景进行交互
  - 创建并渲染具有自定义相机轨迹的场景
  - 查看不同的输出类型
  - 还有更多功能！
- ✏️ 支持多种日志记录接口（Tensorboard、Wandb）、代码性能分析和其他内置调试工具
- 📈 在 Blender 数据集上易于使用的基准测试脚本
- 📱 完整的流水线支持（使用 Colmap、Polycam 或 Record3D），可将手机上的视频转换为完整的 3D 渲染。

# 构建基础

<a href="https://github.com/brentyi/tyro">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://brentyi.github.io/tyro/_static/logo-dark.svg" />
<!-- /pypi-strip -->
    <img alt="tyro logo" src="https://brentyi.github.io/tyro/_static/logo-light.svg" width="150px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- 易于使用的配置系统
- 由 [Brent Yi](https://brentyi.com/) 开发

<a href="https://github.com/KAIR-BAIR/nerfacc">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png" />
<!-- /pypi-strip -->
    <img alt="tyro logo" src="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" width="250px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- 用于加速神经辐射场渲染的库
- 由 [Ruilong Li](https://www.liruilong.cn/) 开发

# 引用

你可以在 [arXiv](https://arxiv.org/abs/2302.04264) 上找到该框架的论文。

如果你使用了这个库或发现文档对你的研究有帮助，请考虑引用：

```
@inproceedings{nerfstudio,
    title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
    author       = {
        Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
        and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
        Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
        Angjoo
    },
    year         = 2023,
    booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
    series       = {SIGGRAPH '23}
}
```

# 贡献者

<a href="https://github.com/nerfstudio-project/nerfstudio/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nerfstudio-project/nerfstudio" />
</a>
```
