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
        self.seg_head[0] = nn.Conv2d(in_channels, 256, kernel_size=1)
        # # 修改最后一层分类输出
        self.seg_head[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        # self.seg_head0 = nn.Conv2d(in_channels, 256, kernel_size=(3,3), padding=1)
        # self.seg_head1 = nn.Conv2d(256, 256, kernel_size=(3,3), padding=1)
        # self.seg_head2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.atv0 = nn.ReLU()
        # self.seg_head3 = nn.Conv2d(256, num_classes, kernel_size=1)
        # self.seg_head = nn.Sequential(self.seg_head0, self.seg_head2, self.atv0, self.seg_head3)

    def forward(self, x):
        features = self.backbone(x)  # (B, 196, 384)
        B, N_patch, D = features.shape
        h = w = int(N_patch ** 0.5)  # 假设是方形patch布局
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


def save_model_checkpoint(model, optimizer, epoch, output_dir, filename="checkpoint.pth"):
    ckpt_path = output_dir 
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, ckpt_path + filename)


from torchvision import transforms
def load_datasets(dataset_str="ADE20K:split=TRAIN", batch_size=16, transform=None, target_transform=None):
    # 用仓库的解析器创建数据集（替换 /path/to/ade20k 为你的 ADE20K 根目录）
    # split 可以等于： TRAIN 或 VAL; 等于VAL时，读取验证集
    ds = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=target_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

# 加载权重初始化模型
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
def load_header_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model" in state_dict:
            model_dict = state_dict["model"]
            missing, unexpected = model.load_state_dict(model_dict, strict=False)
            print("加载model参数成功")
            # 加载优化器参数
        if optimizer is not None and "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
            print("加载optimizer参数成功")
            # 获取上次训练的 epoch
        if "epoch" in state_dict:
            start_epoch = state_dict["epoch"] + 1
        else:
            start_epoch = 0

        # # 去掉可能的 'module.' 前缀（如果是 DataParallel 保存的）
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        logger.info("Loaded pretrained weights.")
    else:
        raise ValueError("Checkpoint path should be a file, not a directory.")
    return model, optimizer, start_epoch

class KLLossForSeg(torch.nn.Module):
    def __init__(self, num_classes, ignore_index=255, reduction='batchmean'):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)  # [N, C, H, W]
        ignore_mask = (target == self.ignore_index)

        target_clamped = target.clone()
        target_clamped[ignore_mask] = 0  # 先不越界
        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        target_one_hot[ignore_mask.unsqueeze(1).expand_as(target_one_hot)] = 0

        loss = F.kl_div(log_probs, target_one_hot, reduction=self.reduction)
        return loss
    
# 计算mIou指标衡量
import numpy as np
def compute_mIoU(preds, targets, num_classes, ignore_index=255):
    """
    计算 mIoU
    preds: (N, H, W) 预测的类别id
    targets: (N, H, W) 真实的类别id
    num_classes: 类别数
    ignore_index: 忽略的标签（如255）
    """
    # 转 numpy
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        if target_inds.sum() == 0:  # 数据里没这个类
            continue

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            iou = float('nan')  # 避免除零
        else:
            iou = intersection / union
        ious.append(iou)

    mIoU = np.nanmean(ious)
    return mIoU

import matplotlib.pyplot as plt
def do_test(image, mask, model, isshow=False,num_classes=151):
    # model.eval()
    model.eval() # 设置为评估模式，关闭 dropout 等
    with torch.no_grad():
        outputs_val = model(image)  # (B, num_classes, H, W)
        logits_val = outputs_val["out"] if isinstance(outputs_val, dict) else outputs_val
        segmentation_result = torch.argmax(logits_val, dim=1)

        out_mask = segmentation_result[1].detach().cpu().numpy()
        # image = images[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
        # mask_ori = masks[0].permute(1, 2, 0).cpu().numpy()
        masks_c1_val = mask[1].cpu().detach().numpy()

        # miou = compute_mIoU(out_mask,masks_c1_val,num_classes=151)
        miou = compute_mIoU(segmentation_result,mask,num_classes=num_classes)
        print("miou:", miou)

        # targets_one_hot = F.one_hot(masks_c1, num_classes=class_num)
        if isshow:
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
    model.train()
        
    return miou

import pandas as pd
import json
import os
def draw_loss_curve(all_loss, epoch, isshow=True, issave=False, name="none"):
    if isshow:
        df = pd.DataFrame(all_loss, columns=["Loss"])
        plt.figure(figsize=(10, 6))
        plt.plot(df["Loss"], label="CrossEntropyLoss", alpha=0.7)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()
        os.makedirs(f"./lossdata/{name}", exist_ok=True)
        plt.savefig(f"./lossdata/{name}/loss_curve_{name}_{epoch + 1}.png")

    if issave:
        # 保存loss数据
        os.makedirs(f"./lossdata/{name}", exist_ok=True)
        with open(f"./lossdata/{name}/training_metrics_loss_{name}.json", "a") as f: # 根目录下的 lossdata 文件夹
            for i, loss_value in enumerate(all_loss):
                json_line = json.dumps({"iteration": i, "total_loss": loss_value})
                f.write(json_line + "\n")

# 绘制miou曲线
def draw_miou_curve(all_miou, epoch, isshow=True, issave=False, name="none"):
    if isshow:
        df = pd.DataFrame(all_miou, columns=["mIoU"])
        plt.figure(figsize=(10, 6))
        plt.plot(df["mIoU"], label="mIoU", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.title("Validation mIoU")
        plt.legend()
        plt.show()
        os.makedirs(f"./mioudata/{name}", exist_ok=True)
        plt.savefig(f"./mioudata/{name}/miou_curve_epoch_{epoch + 1}.png")

    if issave:
        # 保存miou数据
        import json
        os.makedirs(f"./mioudata/{name}", exist_ok=True)
        with open(f"./mioudata/{name}/miou_metrics_epoch_{name}.json", "a") as f:
            for i, miou_value in enumerate(all_miou):
                json_line = json.dumps({"epoch": i * 20 + 1, "mIoU": miou_value})
                f.write(json_line + "\n")

from unetclass import DinoUNet
def main():
    class_num = 151
    segheadname = "deeplab-v1"
    isloadckpt = True # 是否加载权重
    ckptepoch = 600
    iseval = True # 是否为权重测试模式
    height = 224
    width = 224
    start_epoch = 0
    
    backbone = vit_small(
        patch_size=16,
        num_classes=class_num,  # 100 类
        drop_path_rate=0.0
    )
    backbone.eval() # 设置为评估模式，关闭 dropout 等
    load_dinov3_weights(backbone,
                        "./dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                        train=False)
    
    if segheadname.split("-")[0] == "unet":
        model = DinoUNet(backbone=backbone, num_classes=class_num, backbone_out_channels=384,
                    decoder_channels=(384, 256), use_transpose=True, final_upsample=True)
    elif segheadname.split("-")[0] == "deeplab":
        model = DinoDeepLab(backbone, num_classes=class_num)
    else:
        raise ValueError("只支持 unet 分割头")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 3. 定义损失和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 忽略背景类
    # criterion = KLLossForSeg(num_classes=class_num, ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 加载模型预训练权重文件
    if isloadckpt:
        _,_, start_epoch = load_header_checkpoint(model, optimizer, f"./dinov3/ckpt_{segheadname}_imgs64_224p16_{ckptepoch}.pth", device=device)
    # load_header_checkpoint(model, "./dinov3/checkpoint_epoch_10.pth")

    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),   # 保留类别id
        transforms.Lambda(lambda x: x.squeeze().long())  # [1,H,W] -> [H,W]
    ])
    
    loader = load_datasets(dataset_str="ADE20K:split=TRAIN", 
                           batch_size=16, # 一次加载数据集大小
                           transform=transform, 
                           target_transform=target_transform)
    
    # do train
    times = 100
    all_loss = [] # 记录所有的损失值
    all_miou = [] # 记录所有的miou值
    for epoch in range(start_epoch, 901):  
        model.train()
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
            all_loss.append(loss.item())

            if iseval:
                do_test(images, masks_c1, model, isshow=True,num_classes=class_num)

            # # 保留最近的 3 个检查点
            # keep_last_n_checkpoints(".", prefix="checkpoint_epoch_", n=3)

        # 每迭代1次(times)则保存一次检查点
        if (epoch + 1) % times == 0:
            # pass
            if not iseval:
                print(f"Saving checkpoint for epoch {epoch + 1}")
                save_model_checkpoint(model, optimizer, epoch, output_dir="./",filename=f"ckpt_{segheadname}_imgs64_224p16_{epoch+1}.pth")
            miou = do_test(images, masks_c1, model, isshow=iseval, num_classes=class_num)
            all_miou.append(miou)
            # 保存损失函数参数值，并画出曲线
            if not iseval:
                draw_loss_curve(all_loss, epoch, isshow=True, issave=True, name=segheadname)
                draw_miou_curve(all_miou, epoch, isshow=False, issave=True, name=segheadname)

    # --------

    # 转换为 numpy 数组
    # segmentation_result = segmentation_result.squeeze(0).numpy()  # (256, 256)

    # model = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1).cuda()
    # x = torch.randn(1, 3, 224, 224).cuda()
    # x = images[0].unsqueeze(0).cuda()
    # image = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # (H,W,3)

# 函数：从文件读取图像
def load_image(image_path):
    from PIL import Image
    image = Image.open(image_path)
    return image

if __name__ == "__main__":
    main()

    # # PIL图像转换成Tensor图像
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # image_tensor = transform(image).unsqueeze(0).cuda()
    # print("Tensor形状:", image_tensor.shape)
