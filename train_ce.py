import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import numpy as np
from unet import UNet
from tqdm import tqdm  # <-- 引入进度条库


class TargetTransform:
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.long) - 1


def get_dataloaders(batch_size=8):
    # 🌟 核心优化 1：将 256 改为 128，极大降低 CPU 计算量
    IMAGE_SIZE = 128

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        TargetTransform()
    ])

    print("正在检查数据集...")
    train_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', download=False,
                                           transform=transform, target_transform=target_transform)
    val_dataset = datasets.OxfordIIITPet(root='./data', split='test', target_types='segmentation', download=False,
                                         transform=transform, target_transform=target_transform)

    # 确保 num_workers=0 避免 Windows 死锁
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def calculate_miou(preds, targets, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union != 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def main():
    epochs = 15
    batch_size = 8
    lr = 1e-4

    # 禁用严格同步，防止网络波动卡死
    wandb.init(project="unet-pet-segmentation", name="run_loss_ce", settings=wandb.Settings(_disable_stats=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n========================================")
    print(f"当前使用设备: {device} | 正在运行: Cross-Entropy Loss")
    print(f"图像尺寸已优化至 128x128，开启加速模式")
    print(f"========================================\n")

    model = UNet(n_channels=3, n_classes=3).to(device)
    train_loader, val_loader = get_dataloaders(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0

        # 🌟 核心优化 2：使用 tqdm 包装 train_loader
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] 训练中", leave=False, dynamic_ncols=True)

        for images, masks in train_pbar:
            images, masks = images.to(device), masks.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_ce(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            # 实时更新进度条后缀，显示当前 Loss
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss_avg = train_loss_total / len(train_loader)

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_loss_total, val_miou_total = 0.0, 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] 验证中", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.squeeze(1).to(device)
                outputs = model(images)
                loss = criterion_ce(outputs, masks)
                val_loss_total += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_miou_total += calculate_miou(preds, masks)

        val_loss_avg = val_loss_total / len(val_loader)
        val_miou_avg = val_miou_total / len(val_loader)

        # 打印当前 Epoch 的最终结果
        print(
            f"Epoch [{epoch + 1}/{epochs}] 完成 | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val mIoU: {val_miou_avg:.4f}")
        wandb.log({"epoch": epoch + 1, "Train Loss (CE)": train_loss_avg, "Val Loss (CE)": val_loss_avg,
                   "Val mIoU (CE)": val_miou_avg})

    torch.save(model.state_dict(), "unet_ce_weights.pth")
    print("CE 模型已保存！")
    wandb.finish()


if __name__ == '__main__':
    main()