<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_sparseinst/main/icons/sparseinst.png" alt="Algorithm icon">
  <h1 align="center">train_sparseinst</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_sparseinst">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_sparseinst">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_sparseinst/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_sparseinst.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train Sparseinst instance segmentation models.

![Sparseinst instance segmentation baseball game](https://github.com/hustvl/SparseInst/blob/main/assets/figures/000000006471.jpg?raw=true)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")
coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "instance_segmentation",
}) 

# Add training algorithm
train = wf.add_task(name="train_sparseinst", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'sparse_inst_r50_giam_aug': Name of the Sparseinst model. Additional models are available:
    - sparse_inst_r50vd_base
    - sparse_inst_r50_giam
    - sparse_inst_r50_giam_soft
    - sparse_inst_r50_giam_aug
    - sparse_inst_r50_dcn_giam_aug
    - sparse_inst_r50vd_giam_aug
    - sparse_inst_r50vd_dcn_giam_aug
    - sparse_inst_r101_giam
    - sparse_inst_r101_dcn_giam
    - sparse_inst_pvt_b1_giam
    - sparse_inst_pvt_b2_li_giam

- **batch_size** (int) - default '8': Number of samples processed before the model is updated.
- **max_iter** (int) - default '4000': Maximum number of iterations. 
- **eval_period** (int) - default '50': Interval between evaluations.  
- **dataset_split_ratio** (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- **output_folder** (str, *optional*): path to where the model will be saved. 

**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")
coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "instance_segmentation",
}) 

# Add training algorithm
train = wf.add_task(name="train_sparseinst", auto_connect=True)
train.set_parameters({
    "model_name": "sparse_inst_r50vd_base",
    "batch_size": "4",
    "max_iter": "1000",
    "eval_period": "100",
    "dataset_split_ratio": "0.8",
}) 

# Launch your training on your data
wf.run()
```

