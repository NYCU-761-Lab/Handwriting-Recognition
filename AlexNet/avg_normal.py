"""
structure:  AlexNet
pooling:    average
para.pth:   ./avg_normal_params.pth
dilate:     kernel(5, 5)

reference:  
1. AlexNet: https://blog.csdn.net/qq_34644203/article/details/104901786
2. split numbers: https://blog.csdn.net/qq8993174/article/details/89081859
3. network reference:
    (1) https://blog.csdn.net/beilizhang/article/details/114807194
    (2) https://blog.csdn.net/dcrmg/article/details/79241211
"""
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
 
import cv2
import numpy as numpy
import matplotlib.pyplot as plt
# 定義是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Part1: AlexNet
"""
# 定義 AlexNet 網路結構
class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        # 共 8 層：5層卷基層 + 3層全連接層
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential( # 輸入1*28*28 -> *** 原先架構為 224*224，為了訓練 MNIST dataset 修改過大小
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32*28*28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # 32*14*14
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64*14*14
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # 64*7*7
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128*7*7
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256*7*7
            nn.ReLU(inplace=True),
            )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*7*7
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2), # 256*3*3
            # LRN -> not efficient
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.ReLU(inplace=True),  # add
            nn.Dropout(0.5),        # add
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),  # add
            nn.Dropout(0.5),        # add
        )
        self.fc3 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
 
 
# 超參數設置
EPOCH = 10 # 遍歷數據集次數
BATCH_SIZE = 64  # 批處理尺寸(batch_size)
LR = 0.01  # 學習率
 
# 定義數據預處理方式
transform = transforms.ToTensor()
 
# 定義訓練數據集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=False,
    transform=transform)
 
# 定義訓練批處理數據
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
 
# 定義測試數據集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=False,
    transform=transform)
 
# 定義測試批處理數據
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
 
 
# 定義損失函數 loss function 和優化方式(採用 SGD)
net = AlexNet().to(device)
"""*"""
a = torch.load('./avg_normal_params.pth')
net.load_state_dict(torch.load('./avg_normal_params.pth'))
"""*"""
criterion = nn.CrossEntropyLoss()  # 交叉熵損失函數，通常用於多分類問題上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
 
# 訓練並保存模型參數
def train():
 
    for epoch in range(EPOCH):
        # sum_loss = 0.0
        # 數據讀取
        for i, data in enumerate(trainloader):
            # inputs: 輸入 image
            # label:  ground truth 數字
            inputs, labels = data

            # to device: 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
            inputs, labels = inputs.to(device), labels.to(device)
 
            # 梯度清零
            optimizer.zero_grad()
 
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # 反向傳遞 loss
            optimizer.step()# 梯度計算好後，用這個更新參數

            # 每訓練100個batch打印一次平均loss -> disable
            # sum_loss += loss.item()
            # if i % 100 == 99:
            #     print('[%d, %d] loss: %.03f'
            #           % (epoch + 1, i + 1, sum_loss / 100))
            #     sum_loss = 0.0

        # 每跑完一次epoch測試一下準確率(每個 epoch 應該是前後關係，ideal下會希望越學越好)
        with torch.no_grad():   # test data 傳入看結果，不希望影響到 net 的參數也來學這筆資料 -> 變都在 train
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那那個類 -> 數字為多少
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('%d epoch\'s accuracy: %.4f%%' % (epoch + 1, (100 * correct / total)))
        # 保存模型參數
        torch.save(net.state_dict(), './avg_normal_params.pth')
 
"""
    Part2: process detect image
"""
# 反相灰度圖，將黑白閾值顛倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 黑白顛倒 binary 圖像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    kernel = numpy.ones((5, 5), numpy.uint8)
    # img = cv2.erode(img, kernel, iterations = 1) 
    # -> 不能用 erode，東西會出事，主要可能是因為數字較於整張圖片太小，做起來會吞掉很多小數字的細節(?)
    # 邊緣膨脹(用捲積 kernel)，讓比劃更粗(原本真的偏細)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img


# 尋找邊緣，返回邊框的左上角跟右下角（利用cv2.findContours）
def findBorderContours(path, maxArea=50):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img) # find countours 的輸入必須是 binary
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 將邊緣擬合成一個邊框
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > maxArea:
            border = [(x, y), (x+w, y+h)]
            borders.append(border)
    return borders

# 根據邊框轉成MNIST格式
def transMNIST(path, borders, size=(28, 28)):
    imgData = numpy.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根據最大邊擴展窄邊的像素至正方形 28*28
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = numpy.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

def predict(imgData):
    result_ans = []
    for img in imgData:
        img = torch.from_numpy(img).float()
        img = img.view(1, 1, 28, 28)
        img = img.to(device)
        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)
        result_ans.append(predicted.to('cpu').numpy().squeeze())
        # see single number image after processing
        # plt.imshow(img[0][0], cmap='gray', interpolation='none')
        # plt.show()
    return result_ans

# 显示结果及边框
def showResults(path, borders, results=None):
    img = cv2.imread(path)
    # 绘制
    # print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
        #cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('detect', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # train()
    """
    multiple numbers per image
    """
    path = '../image/self_70.png'
    borders = findBorderContours(path)  # 1. find the border for the handwriting numbers on paper
    imgData = transMNIST(path, borders) # 2. transmit those images into single image with 28*28 size
    results = predict(imgData)          # 3. determine the answer for single image
    # print(results)
    showResults(path, borders, results) # 4. show determine result with the original image marked the number


    """
    single number per image
    """
    # img = cv2.imread('./image/gray.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img,(28, 28))
    # print(img)
    # for i in range(0, 28, 1):
    #     for j in range(0, 28, 1):
    #         img[i][j] = 255 - img[i][j]

    # img = torch.from_numpy(img).float()
    # img = img.view(1, 1, 28, 28)

    # img = img.to(device)
    # outputs = net(img)
    # _, predicted = torch.max(outputs.data, 1)
    # print(predicted.to('cpu').numpy().squeeze())
    
    # # show the image
    # plt.imshow(img[0][0], cmap='gray', interpolation='none')
    # plt.show()

 