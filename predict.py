import torch
from PIL import Image
from torchvision import transforms

# GPU加速
is_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if is_gpu else "cpu")

# CIFAR10数据集的类别名
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classToIdx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
              'ship': 8, 'truck': 9}

# 数据变换
transform = transforms.Compose([
    transforms.Resize(size=(232, 232), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predictImage_api(img_path):
    # 读取图片
    img = Image.open(img_path).convert('RGB')
    # 图像处理
    img_Transform = transform(img)
    # 对图像进行升维
    img_Transform = torch.unsqueeze(input=img_Transform, dim=0)
    img_Transform = img_Transform.to(device)
    # 加载模型
    modelResNet50 = torch.load('models/modelResNet50_9.pth')
    modelResNet50 = modelResNet50.to(device)
    # print(modelResNet50)
    # 预测图片
    predicts = modelResNet50(img_Transform)
    # 获取类别概率值
    predicts = torch.nn.functional.softmax(predicts, dim=1)[0]
    class_id = predicts.argmax()
    print('predicts: {}'.format(predicts))
    print('class_id: {}'.format(class_id))
    print('预测概率: {:.4f}'.format(predicts[class_id].item()))
    print('预测类别: {}'.format(classes[class_id]))


if __name__ == '__main__':
    # 预测图片
    predictImage_api(img_path='test_dog4.jpg')
    
