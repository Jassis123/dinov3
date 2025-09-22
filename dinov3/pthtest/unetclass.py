import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    """上采样块：先上采样（ConvTranspose2d 或 interpolate + conv），再 convblock"""
    def __init__(self, in_ch, out_ch, use_transpose=False):
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = ConvBlock(out_ch, out_ch)
        else:
            # 使用插值 + 1x1 调整通道，再 convblock
            self.up = None
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            self.conv = ConvBlock(out_ch, out_ch)
            
    def forward(self, x):
        if self.up is not None:
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = self.proj(x)
        x = self.conv(x)
        return x
    
class DinoUNet(nn.Module):
    """
    DinoUNet: 一个轻量级 UNet-style decoder 以接在 ViT backbone 后。
    - backbone: 接受 (B,3,H,W) -> 输出 (B, N, D) 或 (B, D, H', W')
    - num_classes: 语义类别数
    - backbone_out_channels: D（例如 384）
    - decoder_channels: list, 自顶向下 decoder 每层的通道数（从 bottleneck 到高分辨率）
    - use_transpose: 是否用 ConvTranspose2d 做 upsampling（否则用 bilinear+1x1）
    Note: 默认假设 backbone 最终特征分辨率相对于输入是 1/16（224->14）。
    """
    def __init__(self,
                 backbone,
                 num_classes,
                 backbone_out_channels=384,
                 decoder_channels=(256, 128, 64, 32),
                 use_transpose=False,
                 final_upsample=True):
        super().__init__()
        self.backbone = backbone
        self.backbone_out_channels = backbone_out_channels
        self.decoder_channels = decoder_channels
        self.use_transpose = use_transpose
        self.final_upsample = final_upsample

        # 投影：把 backbone D -> decoder_channels[0]
        self.bottleneck_proj = nn.Conv2d(backbone_out_channels, decoder_channels[0], kernel_size=1)

        # 构造上采样层（从 bottleneck 向上）
        ups = []
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            ups.append(UpBlock(in_ch, out_ch, use_transpose=use_transpose))
            in_ch = out_ch
        self.ups = nn.Sequential(*ups)

        # 最后分类卷积：把最后一个 channel 映射到 num_classes
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _backbone_forward(self, x):
        """调用 backbone，兼容多种输出形式"""
        feat = self.backbone(x) # (B, N,D)
        # 可能输出 (B, N, D) 或 (B, D, H, W) 或 (B, D)。
        if isinstance(feat, tuple) or isinstance(feat, list):
            # 若 backbone 返回多个尺度 (optional)，取最后一个为 bottleneck
            feat = feat[-1]
        if feat.ndim == 2:
            # (B, D) -- 仅 CLS token，不可用于分割
            raise ValueError("Backbone 输出仅为 (B,D) 的 CLS token，无法用于分割，需获取 patch tokens 或使用 forward_features 接口。")
        if feat.ndim == 3:
            # (B, N, D) -> 去掉 cls 若存在（N may be 197），并 reshape
            B, N, D = feat.shape
            # 若包含 cls token 假设第一个是 cls
            if N == 1:
                raise ValueError("Backbone N==1, 无空间信息")
            # 如果 N 是 (H'*W') 或 (1+H'*W')，处理如下：
            if (int(N**0.5))**2 == N:
                # perfect square: no cls token
                h = w = int(N**0.5)
                tokens = feat
            elif (int((N-1)**0.5))**2 == (N-1):
                # first token likely CLS
                h = w = int((N-1)**0.5)
                tokens = feat[:, 1:, :]
            else:
                # 退而求其次：尝试找到近似平方的下采样分辨率
                approx = int((N)**0.5)
                if approx * approx == N:
                    h = w = approx
                    tokens = feat
                else:
                    # 无法确定 H,W
                    raise RuntimeError(f"无法确定 token 空间分辨率 N={N}")
            # reshape -> (B, D, h, w)
            tokens = tokens.permute(0, 2, 1).contiguous()
            feat_map = tokens.view(B, D, h, w)
            return feat_map
        if feat.ndim == 4:
            # 已经是 (B, D, H, W)
            return feat
        raise RuntimeError("无法识别 backbone 输出形状")

    def forward(self, x):
        # x: (B,3,H,W)
        feat_map = self._backbone_forward(x)        # (B, D, h, w)
        # 如果 backbone 的 D 与我们预期不一致，仍可适配（bottleneck_proj handles）
        # x = self.bottleneck_proj(feat_map)         # (B, C0, h, w)
        x = feat_map
        # 顺序上采样
        if len(self.ups) > 0:
            x = self.ups(x)
        # classifier -> (B, num_classes, h_up, w_up)
        out = self.classifier(x)
        # 将输出上采样到输入分辨率（若 final_upsample True）
        if self.final_upsample:
            # out = F.interpolate(out, size=(feat_map.shape[2]* (2**len(self.ups)), feat_map.shape[3]* (2**len(self.ups))),
            #                     mode="bilinear", align_corners=False)
            # 上面 size 写法为估算；通常我们更直接把它恢复成输入 shape：
            orig_H = 224
            orig_W = 224
            out = F.interpolate(out, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
            # out = F.interpolate(x, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
        
        out = self.classifier(out)

        return {"out": out}


# 假设你已有一个 backbone，示例用一个 fake backbone:
class FakeBackbone(torch.nn.Module):
    def __init__(self, D=384, patch_hw=14):
        super().__init__()
        self.D = D
        self.patch_hw = patch_hw
    def forward(self, x):
        B = x.shape[0]
        N = self.patch_hw * self.patch_hw
        # 返回形状 (B, N, D)
        return torch.randn(B, N, self.D, device=x.device)

# # 构建模型
# backbone = FakeBackbone(D=384, patch_hw=14)
# model = DinoUNet(backbone=backbone, num_classes=151, backbone_out_channels=384,
#                  decoder_channels=(384, 256), use_transpose=False, final_upsample=True)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# # 测试 forward
# imgs = torch.randn(2, 3, 224, 224).to(device)
# out = model(imgs)
# print("out keys:", out.keys())
# print("out['out'] shape:", out["out"].shape)  # e.g. (2, num_classes, H_out, W_out)