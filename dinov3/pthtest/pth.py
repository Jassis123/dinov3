import torch
from dinov3.models.vision_transformer import vit_small
from dinov3.data import make_dataset


# 加载 DINOv3 的权重
def load_dinov3_weights(model, checkpoint_path,train=False):
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    if not train: 
        for param in model.parameters(): # 冻结参数
            param.requires_grad = False

    return model

# -----

# # ckpt = torch.load("./dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", map_location="cuda")
# ckpt = torch.load("./dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", map_location="cuda")
# # print(ckpt.keys())
# # print(ckpt["teacher"].keys())  # 看 teacher 里面的内容
# # 有些 ckpt 会包含 "state_dict" 字段
# state_dict = ckpt.get("model", ckpt)

# backbone = vit_small(
#     patch_size=16,
#     num_classes=150,  # 150 类
#     drop_path_rate=0.0
# )
# # 加载参数
# missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
# # print("Missing keys:", missing)
# # print("Unexpected keys:", unexpected)

# for param in backbone.parameters(): # 冻结参数
#     param.requires_grad = False

# -----

# 全连接分类头
# head = torch.nn.Linear(384, 1000)  # 384 是 vit_small 的特征维度
# model = torch.nn.Sequential(backbone, head)
# model = model.cuda().eval() # 作用：转移到 GPU 并设置为评估模式
# # 评估模式的作用：在评估模式下，某些层（如 Dropout 和 BatchNorm）会表现得不同，以便于模型评估
# # 不同在于：训练模式下，Dropout 会随机丢弃一部分神经元，而 BatchNorm 会使用当前批次的统计信息；评估模式下，Dropout 不会丢弃神经元，BatchNorm 会使用整个训练集的统计信息。


import torch.nn as nn
import torchvision
import torch.nn.functional as F
# 替换 DeepLabV3 的 backbone
class DinoDeepLab(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        deeplab = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None, num_classes=num_classes
        )
        self.seg_head = deeplab.classifier

        # 修改输入通道 (默认2048 → 384)
        in_channels = 384
        # self.seg_head[0] = nn.Conv2d(in_channels, 256, kernel_size=1)
        # # # 修改最后一层分类输出
        # self.seg_head[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.seg_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # self.seg_head1 = nn.Conv2d(256, 2048, kernel_size=(3,3))
        # self.seg_head2 = nn.Conv2d(2048, num_classes, kernel_size=1)
        # self.seg_head = nn.Sequential(self.seg_head0, self.seg_head1, self.seg_head2)

    def forward(self, x):
        B, N_patch, D = features.shape
        h = w = int(N_patch ** 0.5)  # 假设是方形patch布局

        features = self.backbone(x)  # (B, 196, 384)
        features = features.permute(0, 2, 1).reshape(B, D, h, w)  # (B, 384, 14, 14)
        # features = features.reshape(x.shape[0], 384, 14, 14)  # (B, D, 1, 1)

        out = self.seg_head(features)  # (B, num_classes, 14, 14)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": out}
    
        # if isinstance(features, torch.Tensor):  
        #     # 如果是 [CLS] token，则需要 reshape
        #     raise ValueError("Backbone输出需要改为特征图")  
        # 否则 features 应该是 (B, C, H, W)
        # return {"out": self.seg_head(features)}
    
# model = DinoDeepLab(backbone, num_classes=150)  # COCO 有 150 类
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# # 直接用现有的 dinov3.pth 做 语义分割训练（Linear Probe + Fine-tune 两种模式）
# x = torch.randn(2, 3, 224, 224).cuda()
# with torch.no_grad():
#     logits = model(x)  # (2, 1000)
# print(logits.shape)  # 应该是 (2, 1000)


# ------
# import matplotlib.pyplot as plt

# # 假设模型输出的维度是 (batch_size=1, num_classes=21, height=256, width=256)
# batch_size = 1
# num_classes = 21
# height = 256
# width = 256

# # 模拟模型输出 (随机生成概率图)
# model_output = torch.rand(batch_size, num_classes, height, width)

# # 使用 argmax 找到每个像素的类别索引
# # 输出维度变为 (batch_size=1, height=256, width=256)
# segmentation_result = torch.argmax(model_output, dim=1)

# # 转换为 numpy 数组
# segmentation_result = segmentation_result.squeeze(0).numpy()  # (256, 256)

# # 可视化分割结果
# plt.imshow(segmentation_result, cmap='jet')  # 使用颜色映射显示分割结果
# plt.colorbar()
# plt.show()
# ------

# from torchvision.datasets import CocoDetection
# import torchvision.transforms as T
# from torch.utils.data import DataLoader

# train_transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
# ])

# target_transform = T.Compose([
#     T.Resize((224, 224)),
#     # 语义分割 mask 保持整数 ID
# ])

# import os
# cwd = os.getcwd()
# root = os.path.dirname(cwd)   # 上一级目录

# trainset = CocoDetection(
#     root=os.path.join(root, "val2017/val2017"),
#     annFile=os.path.join(root, "annotations_trainval2017", "captions_train2017.json"),
#     transform=train_transform,
#     target_transform=target_transform,
# )

# trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

def save_model_checkpoint(model, optimizer, epoch, output_dir, filename="checkpoint.pth"):
    ckpt_path = output_dir 
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, ckpt_path + filename)

# ------

# # 用仓库的解析器创建数据集（替换 /path/to/coco 为你的 COCO 根目录）
# #split可以等于： TRAIN 或 VAL; 等于VAL时，读取验证集
# from torchvision import transforms
# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# ds = make_dataset(dataset_str="CocoCaptions:split=VAL", transform=transform)
# loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

# for images, targets in loader:
#     images = images.cuda()
#     with torch.no_grad():
#         logits = model(images)  # (B, num_classes, H, W)
#     break

# --- ADE20K 数据集 ---

from torchvision import transforms
def load_datasets(dataset_str="ADE20K:split=TRAIN", batch_size=16, transform=None, target_transform=None):
    # 用仓库的解析器创建数据集（替换 /path/to/ade20k 为你的 ADE20K 根目录）
    # split 可以等于： TRAIN 或 VAL; 等于VAL时，读取验证集
    ds = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=target_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


# from torchvision import transforms
# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# mask_transform = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
#     transforms.PILToTensor(),   # 保留类别id
#     transforms.Lambda(lambda x: x.squeeze().long())  # [1,H,W] -> [H,W]
# ])
# ds = make_dataset(dataset_str="ADE20K:split=TRAIN", transform=transform, target_transform=mask_transform)
# loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

# # 3. 定义损失和优化器
# criterion = nn.CrossEntropyLoss(ignore_index=255)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# for images, masks in loader:
#     images, masks = images.cuda(), masks.cuda()
#     masks = masks[:,0,:,:].long()
#     outputs = model(images)
#     # 有些模型返回 {"out": tensor, "aux": tensor}，只取 "out"
#     logits = outputs["out"] if isinstance(outputs, dict) else outputs
#     loss = criterion(logits, masks)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# --- 自定义加载数据集 ---
# from torch.utils.data import DataLoader
# from dinov3.data.datasets.coco_instance import CocoInstanceDataset
# from torchvision import transforms
# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# root = "E:/PRJECTSPACE/OPENSOURCEPRJSPACE/DINOV3/"
# val_dataset = CocoInstanceDataset(root=root, split="val", transform=transform)
# loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0,collate_fn=lambda x: tuple(zip(*x)))

# for images, targets in loader:
#     # images = images.cuda()
#     with torch.no_grad():
#         logits = model(images)  # (B, num_classes, H, W)
#     break
# --------

# import matplotlib.pyplot as plt
# segmentation_result = torch.argmax(logits['out'], dim=1)

# # 转换为 numpy 数组
# # segmentation_result = segmentation_result.squeeze(0).numpy()  # (256, 256)

# # 可视化分割结果
# mask = segmentation_result[0].detach().cpu().numpy()
# image = images[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)

# # model = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1).cuda()
# # x = torch.randn(1, 3, 224, 224).cuda()
# # x = images[0].unsqueeze(0).cuda()
# # image = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # (H,W,3)

# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")

# plt.subplot(1, 2, 2)
# plt.imshow(mask, cmap='jet', alpha=0.6)
# plt.title("Segmentation")
# plt.colorbar()
# plt.show()

import matplotlib.pyplot as plt
def main():
    class_num = 151
    # load_checkpoint(model, "./dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
    backbone = vit_small(
        patch_size=16,
        num_classes=class_num,  # 100 类
        drop_path_rate=0.0
    )
    load_dinov3_weights(backbone,
                        "./dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                        train=False)
    model = DinoDeepLab(backbone, num_classes=class_num)  # COCO 有 100 类
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),   # 保留类别id
        transforms.Lambda(lambda x: x.squeeze().long())  # [1,H,W] -> [H,W]
    ])
    
    loader = load_datasets(dataset_str="ADE20K:split=TRAIN", 
                           batch_size=16,
                           transform=transform, 
                           target_transform=target_transform)
    
    # 3. 定义损失和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # do train
    times = 2
    for epoch in range(10):  # 训练2个epoch
        # model.train()
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            masks_c1 = masks.long()
            outputs = model(images)
            # 有些模型返回 {"out": tensor, "aux": tensor}，只取 "out"
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            # # 保留最近的 3 个检查点
            # keep_last_n_checkpoints(".", prefix="checkpoint_epoch_", n=3)

            # with torch.no_grad():  # 不会跟随计算图反向传播影响性能
            #     for name, param in model.named_parameters():
            #         if "conv" in name:
            #             print(f"Layer: {name}")
            #             print(f"Weights: {param}")
            #             print(f"Gradients: {param.grad}")

        # 可视化分割结果
        with torch.no_grad():
            outputs_val = model(images)  # (B, num_classes, H, W)
            logits_val = outputs_val["out"] if isinstance(outputs_val, dict) else outputs_val
            segmentation_result = torch.argmax(logits_val, dim=1)

            out_mask = segmentation_result[0].detach().cpu().numpy()
            # image = images[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
            # mask_ori = masks[0].permute(1, 2, 0).cpu().numpy()
            masks_c1_val = masks_c1[0].cpu().detach().numpy()

        plt.subplot(1, 2, 1)
        # plt.imshow(masks)
        plt.imshow(masks_c1_val, cmap='jet')
        # plt.imshow(image)
        # plt.imshow(mask_ori[:,:,1], cmap='jet')
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(out_mask, cmap='jet')
        # plt.imshow(mask_ori[:,:,0], cmap='jet')
        # plt.imshow(mask_ori)
        plt.title("Segmentation")
        plt.colorbar()
        plt.show()

        # 每迭代1次(times)则保存一次检查点
        if (epoch + 1) % times == 0:
            save_model_checkpoint(model, optimizer, epoch, output_dir="./",filename=f"checkpoint_epoch_{epoch+1}.pth")


    # --------

    # 转换为 numpy 数组
    # segmentation_result = segmentation_result.squeeze(0).numpy()  # (256, 256)



    # model = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1).cuda()
    # x = torch.randn(1, 3, 224, 224).cuda()
    # x = images[0].unsqueeze(0).cuda()
    # image = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # (H,W,3)



if __name__ == "__main__":
    main()

# # Linear Probe: 冻结 backbone
# for p in model.backbone.parameters():
#     p.requires_grad = False



