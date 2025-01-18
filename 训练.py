import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm 库

# 检查是否可以使用 GPU
if not torch.cuda.is_available():
    print("当前环境不支持 GPU 训练，程序将停止。")
    exit()  # 停止程序
else:
    print("GPU 可用，将使用 GPU 进行训练。")

# 数据集路径和模型保存路径
dataset_path = r"C:\Users\GUOXI\Desktop\kunchong\dataset"
model_save_path = r"C:\Users\GUOXI\Desktop\kunchong\models\insect_classifier.pth"

# 调整 batch size
batch_size = 512  # 修改此处以加快训练速度

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 训练集和验证集
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 搭建卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 根据输入图像尺寸调整
        self.fc2 = nn.Linear(128, len(train_dataset.classes))  # 分类数

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加入简单的注意力机制
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def forward(self, x):
        attention_weights = F.softmax(x, dim=1)  # 计算注意力权重
        return x * attention_weights  # 逐元素相乘

# 整合模型
class InsectClassifier(nn.Module):
    def __init__(self):
        super(InsectClassifier, self).__init__()
        self.cnn = SimpleCNN()
        self.attention = AttentionLayer()  # 添加注意力层

    def forward(self, x):
        x = self.cnn(x)
        x = self.attention(x)  # 在卷积后的特征上应用注意力机制
        return x

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InsectClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=500):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # 动态进度条设置
        with tqdm(total=len(train_loader), dynamic_ncols=True, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=True) as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))  # 显示当前平均损失
                pbar.update(1)  # 更新进度条

        print(f'Loss: {running_loss / len(train_loader):.4f}')

# 验证模型
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(val_loader), dynamic_ncols=True, desc='Validating', leave=True) as pbar:
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)  # 更新进度条
        print(f'Accuracy of the model on the validation set: {100 * correct / total:.2f}%')

# 执行训练和验证
train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, val_loader)

# 保存模型
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
