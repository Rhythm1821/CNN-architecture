import torch
from torch import nn

class Lenet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                        out_channels=6,
                        kernel_size=5,
                        stride=1),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,
                            stride=1)
                                     )
        
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      stride=1,
                      kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,
                         stride=2)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      stride=1,
                      kernel_size=5)
        )

        self.layer_4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120,
                      out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84,
                      out_features=10),
            nn.Softmax()
        )
    
    def forward(self,x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        return x

if __name__=="__main__":
    print("Implementing Lenet model.....")
    Lenet()
    print("Lenet model called successully....")