# from transformers import pipeline
# from transformers.image_utils import load_image

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# image = load_image(url)

# feature_extractor = pipeline(
#     model="facebook/dinov3-convnext-small",
#     task="image-feature-extraction",
# )
# features = feature_extractor(image)
# print(features.shape)

import torch
import torch.nn as nn
import os
# import timm

# 1. 加载模型骨干（ViT-7B/16）
# timm 中并没有直接的 vit7b16，但 DINOv3 提供了官方模型定义
# 如果你用的是官方 dinov3 仓库，需要 import 对应的 VisionTransformer
# 这里先假设你有 dinov3 的模型文件：
from dinov3.models.vision_transformer import vit_large,vit_small

# 创建模型骨干 (linear head 意味着最后有个线性分类层)
model = vit_small(
    patch_size=16,
    num_classes=1000,  # imagenet1k
    drop_path_rate=0.0
)

# 2. 加载权重
current_dir = os.path.dirname(os.path.abspath(__file__))
pth_path = os.path.join(current_dir, "dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth")
checkpoint = torch.load(pth_path, map_location="cuda")

# 有些 ckpt 会包含 "state_dict" 字段
state_dict = checkpoint.get("model", checkpoint)

# 加载参数
msg = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", msg.missing_keys)
print("Unexpected keys:", msg.unexpected_keys)

# 3. 测试输入
x = torch.randn(2, 3, 224, 224)  # batch=1, RGB, 224x224
with torch.no_grad():
    y = model(x)
print(y.shape)  # [1, 1000]
