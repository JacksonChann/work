import torch
from torchvision.models import resnet50
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# 读取类别文件
def readClasses():
    classes = []
    with open('imageNet_1.txt', 'r') as fp:
        readlines = fp.read().strip().split('\n')
        for line in readlines:
            classes.append(line[line.find(' ') + 1:].split(',')[0])
    return classes


def Model():
    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if is_gpu else "cpu")

    modelResNet50 = resnet50(pretrained=True, progress=True)

    fc_features = modelResNet50.fc.in_features
    modelResNet50.fc = torch.nn.Linear(fc_features, 10)

    transform = transforms.Compose([
        transforms.Resize(size=(232, 232), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainDataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    testDataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # 提取模型的layer4,avgpool,fc,Classifer层的参数，其他层冻结
    for name, child in modelResNet50.named_children():
        if name in ['layer4', 'avgpool', 'fc']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    batch_size = 4
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                 drop_last=False)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                drop_last=False)

    lossfunction = torch.nn.CrossEntropyLoss()

    learing_rate = 0.001
    optimizer = torch.optim.SGD([
        {'params': modelResNet50.layer4.parameters(), 'lr': learing_rate},
        {'params': modelResNet50.avgpool.parameters(), 'lr': learing_rate},
        {'params': modelResNet50.fc.parameters(), 'lr': learing_rate}
    ], lr=learing_rate)

    # #迭代的次数
    epoches = 10

    for i in range(epoches):
        # 训练的次数
        trainStep = 0
        # 训练集上的准确率
        trainCorrect = 0
        # 训练集上的损失
        trainLoss = 0
        traintotal = 0
        print('——————————第{}轮的训练——————————'.format(i + 1))
        # 每一次迭代的训练
        modelResNet50 = modelResNet50.to(device)
        modelResNet50.train()
        for traindata in trainDataLoader:
            imgs, targets = traindata
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 输入模型，进行训练
            output = modelResNet50(imgs)
            output = output.to(device)
            # 计算损失值
            loss = lossfunction(output, targets).to(device)
            # 优化器清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 步数加一
            trainStep = trainStep + 1
            with torch.no_grad():
                # 获取预测概率最大值索引(按行计算最大)
                output = torch.argmax(input=output, dim=1)
                # 计算准确率
                trainCorrect += (output == targets).sum().item()
                traintotal += imgs.shape[0]
                # 计算总的损失
                trainLoss += loss.item()
                if trainStep % 100 == 0:
                    print('step: {}————loss: {:.6f}————correct: {:.6f}'.format(trainStep,
                                                                               trainLoss * 1.0 / traintotal,
                                                                               trainCorrect * 1.0 / traintotal))

        # 在测试数据集上进行测试
        testTotal = 0
        testCorrect = 0
        testLoss = 0
        # 测试集上不需要梯度更新
        modelResNet50.eval()
        with torch.no_grad():
            for data in testDataLoader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                # 输入模型，进行训练
                output = modelResNet50(imgs)
                output = output.to(device)
                # 计算损失值
                loss = lossfunction(output, targets).to(device)
                # 获取预测概率最大值索引(按行计算最大)
                output = torch.argmax(input=output, dim=1)
                # 计算准确率
                testCorrect += (output == targets).sum().item()
                testTotal += imgs.shape[0]
                # 计算总的损失
                testLoss += loss.item()
            print('-----------test:--------loss: {:.6f}---------correct: {:.6f}'.format(testLoss * 1.0 / testTotal,
                                                                                        testCorrect * 1.0 / testTotal))
        # 保存模型
        torch.save(modelResNet50, './models/modelResNet50_{}.pth'.format(i))


if __name__ == '__main__':
    # readClasses()
    classes = readClasses()
    Model()
