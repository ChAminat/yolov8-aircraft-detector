import torch


class ConvClassifier(torch.nn.Module):
    
    def __init__(self, num_classes: int = 25, in_channels: int = 1):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            # Conv1
            torch.nn.Conv2d(in_channels, 32, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            
            # Conv2
            torch.nn.Conv2d(32, 64, 5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            
            # Conv3
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            
            # Conv4
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            
            # Dropout
            torch.nn.Dropout(0.1),
            
            # Conv5
            torch.nn.Conv2d(256, 512, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            
            # Classifier
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 4 * 4, 256),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)