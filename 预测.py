import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置支持中文的字体，可以根据你的系统调整字体名称
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 其他代码保持不变


# 数据集路径和模型保存路径
dataset_path = r"C:\Users\GUOXI\Desktop\kunchong\dataset"
model_save_path = r"C:\Users\GUOXI\Desktop\kunchong\models\insect_classifier.pth"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 载入类别信息
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)

# 定义模型
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

# 整合模型
class InsectClassifier(nn.Module):
    def __init__(self):
        super(InsectClassifier, self).__init__()
        self.cnn = SimpleCNN()

    def forward(self, x):
        return self.cnn(x)

# 调用模型进行预测
def predict_image(model, image_path):
    model.eval()
    # 打开图像并确保它是RGB格式
    image = Image.open(image_path).convert('RGB')  # 确保转换为RGB
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)  # 计算输出概率
        _, predicted = torch.max(outputs, 1)
        class_name = train_dataset.classes[predicted.item()]
        confidence = probabilities[0][predicted.item()].item() * 100  # 置信度百分比
        
        # 显示预测结果和图像
        plt.imshow(image)
        plt.title(f'预测结果: {class_name} (准确度: {confidence:.2f}%)')  # 显示预测类别和置信度
        plt.axis('off')  # 不显示坐标轴
        plt.show()  # 显示图像

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InsectClassifier().to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))

# 示例：预测一张新图像
image_path = r"C:\Users\GUOXI\Desktop\kunchong\dataset\test\Corn Borers\Image_28.jpg"

# 确保图像文件存在
if not os.path.isfile(image_path):
    print(f"Error: The file {image_path} does not exist.")
    exit(1)

predict_image(model, image_path)
