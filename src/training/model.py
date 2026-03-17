# finetuning the YOLO model on my dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np

# Define a custom dataset class
class YOLODataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = self.annotations[idx]

        if self.transform:
            image = self.transform(image)

        return image, annotation

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

# Load dataset
train_images = r"C:\Users\rutik\Fruit-Ninja\dataset\train"
train_annotations = r"C:\Users\rutik\Fruit-Ninja\dataset\train2"
train_dataset = YOLODataset(train_images, train_annotations, transform=transform)
#val_dataset = YOLODataset(val_images, val_annotations, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo11s.pt')

# Modify the model for the number of classes
model.model[-1].nc = 2  # Number of classes

# Fine-tune the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation loop
    """
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            # Calculate validation metrics
            """

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_yolov11.pt')
