import torch
from ultralytics import YOLO
from torch import nn

# Define a simple EDSR super-resolution module
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, layer):
        residual = self.relu(self.conv1(layer))
        residual = self.conv2(residual)
        return layer + residual

class EDSR(nn.Module):
    """
    A simplified EDSR model for 2x super-resolution.
    """
    def __init__(self, in_channels=3, num_features=64, num_blocks=5, scale=2):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, layer):
        layer = self.head(layer)
        layer = self.body(layer)
        layer = self.tail(layer)
        return layer
    
class YOLOv8SR(YOLO):   
    """

    YOLOv8SR: Customized YOLO model for both object detection with super-resolution.
    
    This class inherits from the Ultralytics YOLO model and extends it by adding
    an EDSR super-resolution module. It's suitable for scenarios where enhanced image
    resolution is crucial along with object detection, such as satellite imagery,
    medical imaging, or high-quality surveillance.

    """

    def __init__(self, cfg='train.yaml', pretrained_weights = "yolov8s.pt"):
        # Initialize the base YOLO model
        super().__init__(pretrained_weights)
        
        # Define EDSR module
        self.edsr = EDSR()
           
    def forward(self, layer):
        # Apply EDSR to enhance input resolution
        layer = self.edsr(layer)
        
        # Pass through YOLOv8 model
        layer = super().forward(layer)
        return layer
    

if __name__ == "__main__":
    # Load pre-trained weights (ensure the model architecture matches the weights)
    pretrained_weights = 'yolov8s.pt'
    cfg='train.yaml'
    # Create an instance of the customYOLOv8  model
    yolov8sr_model = YOLOv8SR(cfg=cfg, pretrained_weights = pretrained_weights)
    
