import torch
import torch.nn as nn


# 1. 定义基础组件：双重卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 2. 搭建完整的 U-Net 模型
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        """
        n_channels: 输入图片的通道数，RGB 图片为 3
        n_classes: 分割的类别数，Oxford-IIIT Pet 为 3 类（前景、背景、边缘）
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- 下采样编码器 (Encoder) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- 上采样解码器 (Decoder) ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # --- 最终输出层 ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码过程
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 解码过程 + Skip Connection
        x = self.up1(x4)
        # 拼接 (concatenate): 在通道维度(dim=1)上将深层和浅层特征合并
        x = self.conv1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits


# 3. 简单的连通性测试
if __name__ == '__main__':
    dummy_input = torch.randn(2, 3, 256, 256)

    model = UNet(n_channels=3, n_classes=3)

    output = model(dummy_input)

    print("模型测试成功！")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")