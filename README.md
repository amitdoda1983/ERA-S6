# ERA-S6


### Part : 2

Below is the updated model network :

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3) #input -? OUtput? RF
        self.conv1_bn=nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv2_bn=nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(p=0.01)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv3_bn=nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 64, 3)
        self.conv4_bn=nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 10, 1)
        self.avgpool = nn.AvgPool2d(3)

    def forward(self, x):
        x=self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.conv3_bn(x))
        x = self.pool2(x)
        x = self.conv4(x)
        x = F.relu(self.conv4_bn(x))
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)

        return F.log_softmax(x)

![image](https://github.com/amitdoda1983/ERA-S6/assets/37932202/51fdf470-af9a-4dfa-b36c-c5cff441854b)

### We have a total of 13,706 parameters

The concepts explored while coming up with this choice of network are:

1. Pyramid style choice of number of kernels or resulting channels
2. use of only 3x3 kernels for expansion
3. use of 1x1 for channel reduction
4. Max pool layers to reduce the channel size.
5. No padding, no stride.
6. Small dropout to generalize.
7. Batch Normalailzation to converge faster.
8. Smaller batch size = 16 to have relatively more weight updates in an epoch


