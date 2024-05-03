import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
import glob
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

CLASS_NUM = 10
EPOCH_NUMBER = 33

# early_stoppongの実装
# modelを変える
# データの前処理を変える(RandomCropやRandomRotationなど)
# データの正規化を変える
# スケジューリングの追加

# trainもaccuracy求めたら面白いかも


class Datasets(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_paths = sorted(glob.glob(img_dir))
        self.label_paths = sorted(glob.glob(label_dir))
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        # original_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        include = torch.tensor([0.0] * CLASS_NUM)
        with open(self.label_paths[index], 'r') as file:
            for line in file:
                A, B, C, E, F = map(float, line.strip().split())
                include[int(A)] = 1.0

        # img = transforms.ToPILImage()(img)
        return img, include, self.img_paths[index]

    def __len__(self):
        return len(self.img_paths)


class MultiLabelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.affine = nn.Linear(24 * 25 * 75, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.affine(x)
        return x


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    out = 0
    right_perfect, total_perfect = 0, 0
    true_positive, false_negative = 0, 0
    false_positive, true_negative = 0, 0
    accuracy, precision, recall, f1 = 0, 0, 0, 0
    for batch_idx, (x, y, z) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        p_y_hat = model(x)

        y_pred_judge = torch.where(p_y_hat >= 0.5, 1.0, 0.0)
        # 完全一致の精度の計算
        if torch.equal(y, y_pred_judge):
            right_perfect += 1
        total_perfect += 1
        # 混同行列の計算
        true_positive += torch.sum((y == 1) & (y_pred_judge == 1)).item()
        false_negative += torch.sum((y == 1) & (y_pred_judge == 0)).item()
        false_positive += torch.sum((y == 0) & (y_pred_judge == 1)).item()
        true_negative += torch.sum((y == 0) & (y_pred_judge == 0)).item()

        loss = F.binary_cross_entropy_with_logits(p_y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch={epoch + 1}, Batch={batch_idx + 1:03}, Loss={loss.item():.4f}")
            out = loss.item()
    # 評価指標の計算
    accuracy = (true_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)
    precision = true_positive / (true_positive + false_positive)  # 予測を正例としたもののうち、実際に正例であるものの割合
    recall = true_positive / (true_positive + false_negative)  # 真に正例のもののうち、正しく正例と予測されたものの割合
    f1 = 2 * precision * recall / (precision + recall)
    writer.add_scalar('Accuracy/train', accuracy, epoch + 1)
    writer.add_scalar('Precision/train', precision, epoch+ 1)
    writer.add_scalar('Recall/train', recall, epoch+ 1)
    writer.add_scalar('F1/train', f1, epoch+ 1)
    writer.add_scalar('Loss/train', loss, epoch+ 1)
    return out  # Lossを返せるようになっている


def val(model, device, data_loader, epoch, writer, print_output=False):
    model.eval()
    right_perfect, total_perfect = 0, 0
    true_positive, false_negative = 0, 0
    false_positive, true_negative = 0, 0
    accuracy, precision, recall, f1 = 0, 0, 0, 0
    for x, y, z in data_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred_judge = torch.where(y_pred >= 0.5, 1.0, 0.0)  # y_pred_judgeがバッチ数*チャネル数で0と1が入っている
            # 完全一致の精度の計算
            right_perfect += torch.sum(torch.all(y == y_pred_judge, dim=1) == 1).item()
            total_perfect += len(y)
            # 混同行列の計算
            true_positive += torch.sum((y == 1) & (y_pred_judge == 1)).item()
            false_negative += torch.sum((y == 1) & (y_pred_judge == 0)).item()
            false_positive += torch.sum((y == 0) & (y_pred_judge == 1)).item()
            true_negative += torch.sum((y == 0) & (y_pred_judge == 0)).item()
            loss = F.binary_cross_entropy_with_logits(y_pred, y)
            if print_output:
                print(z, y, y_pred_judge)
    # 評価指標の計算
    accuracy = (true_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)
    precision = true_positive / (true_positive + false_positive)  # 予測を正例としたもののうち、実際に正例であるものの割合
    recall = true_positive / (true_positive + false_negative)  # 真に正例のもののうち、正しく正例と予測されたものの割合
    f1 = 2 * precision * recall / (precision + recall)
    writer.add_scalar('Accuracy/val', accuracy, epoch + 1)
    writer.add_scalar('Precision/val', precision, epoch + 1)
    writer.add_scalar('Recall/val', recall, epoch + 1)
    writer.add_scalar('F1/val', f1, epoch + 1)
    writer.add_scalar('Loss/val', loss.item(), epoch + 1)
    print(f"val-set perfect accuracy={right_perfect / total_perfect:.04f}")
    print(f"val-set accuracy={accuracy:.04f}, precision={precision:.04f},"
          f" recall={recall:.04f}, f1={f1:.04f}\n")
    return


def main():
    writer = SummaryWriter()
    train_img_dir = "connect/train/images/*.jpg"
    train_label_dir = "connect/train/labels/*.txt"
    val_img_dir = "connect/val/images/*.jpg"
    val_label_dir = "connect/val/labels/*.txt"
    data_transform = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 300), antialias=True),
            # transforms.RandomRotation(5),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 300), antialias=True),
        ])}

    train_dataset = Datasets(train_img_dir, train_label_dir, transform=data_transform['train'])
    val_dataset = Datasets(val_img_dir, val_label_dir, transform=data_transform['val'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    images, labels, _ = next(iter(train_dataloader))
    writer.add_graph(MultiLabelNet().to(device), images.to(device))
    for epoch in range(EPOCH_NUMBER):
        train(model, device, train_dataloader, optimizer, epoch, writer)
        if epoch % 4 == 0:
            val(model, device, val_dataloader, epoch, writer)
        # lr.step()
    val(model, device, val_dataloader, EPOCH_NUMBER, writer, print_output=False)


if __name__ == "__main__":
    main()





# a = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]])
# b = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 0]])
# print(torch.sum(a == b).item())
# print(torch.equal(a, b))


# # aとbの両方が1である要素を見つける
# both_one = (a == 1) & (b == 1)
# print(both_one)
#
# # 1の要素の数を数える
# count = both_one.sum().item()
#
# print(count)

# a = torch.tensor([[0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 0.1]])
# b = torch.where(a >= 0.5, 1.0, 0.0)
# print(a, b)


# a = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 0]])
# b = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 0]])
#
#
# result = torch.all(a == b, dim=1)
# print(torch.sum(torch.all(a == b, dim=1) == 1).item())