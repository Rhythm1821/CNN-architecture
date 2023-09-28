from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Linear(in_features=(256 * 6 * 6),out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=10),
            nn.Softmax()
        )

    def forward(self,x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x
    


AlexNet()