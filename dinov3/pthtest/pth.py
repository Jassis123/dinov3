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

        # self.seg_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # self.seg_head1 = nn.Conv2d(256, 2048, kernel_size=(3,3))
        # self.seg_head2 = nn.Conv2d(2048, num_classes, kernel_size=1)
        # self.seg_head = nn.Sequential(self.seg_head0, self.seg_head1, self.seg_head2)

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
def load_header_checkpoint(model, checkpoint_path):
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights.")
    else:
        raise ValueError("Checkpoint path should be a file, not a directory.")

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
    

import matplotlib.pyplot as plt
def main():
    class_num = 151
    # load_checkpoint(model, "./dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
    backbone = vit_small(
        patch_size=16,
        num_classes=class_num,  # 100 类
        drop_path_rate=0.0
    )
    backbone.eval() # 设置为评估模式，关闭 dropout 等
    load_dinov3_weights(backbone,
                        "./dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                        train=False)
    # 打印backbone参数
    # for name, param in backbone.named_parameters():
    #     print(f"Layer: {name}")
    #     print(f"Weights: {param}")
    #     print(f"Gradients: {param.grad}")
    
    model = DinoDeepLab(backbone, num_classes=class_num)  # COCO 有 100 类
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 加载模型预训练权重文件
    load_header_checkpoint(model, "./dinov3/checkpoint_epoch_imgs64_224p16_400.pth")

    height = 224
    width = 224
    transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),   # 保留类别id
        transforms.Lambda(lambda x: x.squeeze().long())  # [1,H,W] -> [H,W]
    ])
    
    loader = load_datasets(dataset_str="ADE20K:split=TRAIN", 
                           batch_size=16,
                           transform=transform, 
                           target_transform=target_transform)
    
    # 3. 定义损失和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 忽略背景类
    # criterion = KLLossForSeg(num_classes=class_num, ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # do train
    times = 100
    for epoch in range(401):  # 训练2个epoch
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

                out_mask = segmentation_result[1].detach().cpu().numpy()
                # image = images[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
                # mask_ori = masks[0].permute(1, 2, 0).cpu().numpy()
                masks_c1_val = masks_c1[1].cpu().detach().numpy()

                # targets_one_hot = F.one_hot(masks_c1, num_classes=class_num)

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
            
            save_model_checkpoint(model, optimizer, epoch, output_dir="./",filename=f"checkpoint_epoch_imgs64_224p16_{epoch+1}.pth")


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
    # mask = load_image("E:\\PrjectSpace\\OpenSourcePrjSpace\\Dinov3\\ADEChallengeData2016\\annotations\\training\\ADE_train_00000003.png")
    # img = load_image("E:\\PrjectSpace\\OpenSourcePrjSpace\\Dinov3\\ADEChallengeData2016\\images\\training\\ADE_train_00000003.jpg")
    # # img.show()
    # # print(mask.size)  # (宽,高)
    # # reshape图像
    # # img = img.resize((512, 512))
    # # 打印最大值和最小值
    # mask_tensor = torch.tensor(list(mask.getdata()))
    # print("Max:", mask_tensor.max())
    # print("Min:", mask_tensor.min())
    # # 打印图像像素值矩阵
    # # print(torch.tensor(list(img.getdata())))  # torch.Size([512*512, 3])
    # plt.imshow(mask, cmap='jet')
    # plt.colorbar()
    # plt.show()
    # print(list(img.getdata())[:10])  # 打印前10个像素
    

# # Linear Probe: 冻结 backbone
# for p in model.backbone.parameters():
#     p.requires_grad = False



