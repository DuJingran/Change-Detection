# models/__init__.py — 模型工厂函数

"""
    替换新模型步骤：
      1. 在 models/ 下新建 your_model.py
      2. 在此函数中 import 并添加一条 elif
      3. 修改 config.model_name 即可
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.siamunet_conc import SiamUnet_conc
from models.siamunet_diff import SiamUnet_diff  
from models.fresunet import FresUNet


#  模型工厂函数

def build_model(model_name, in_channels, num_classes):

    name = model_name.lower()
    if name == "siamunet_conc":
        return SiamUnet_conc(input_nbr=in_channels, label_nbr=num_classes)
    elif name == "siamunet_diff":
        return SiamUnet_diff(input_nbr=in_channels, label_nbr=num_classes)
    elif name == "ef_unet":
        return FresUNet(input_nbr=in_channels*2, label_nbr=num_classes)
    else:
        raise ValueError(
            f"未知模型名 '{model_name}'。"
            f"请在 models/siamunet.py 的 build_model() 中注册新模型。"
        )