import os
import csv
# from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import crop, to_tensor
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MyModel(nn.Module) :
    def __init__(self,in_channels=3,out_channels=80):
        super().__init__()
        #TODO: Make your own model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1_1 = nn.Conv2d(self.in_channels, 64, kernel_size = 3, padding = 1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding = 1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.fc1 = nn.Linear(512, 2048, bias = True)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2048, self.out_channels, bias=True)
        self.relu = nn.ReLU()
        self.maxPool2d = nn.MaxPool2d(2, stride=2)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm128 = nn.BatchNorm2d(128)
        self.norm256 = nn.BatchNorm2d(256)
        self.norm512 = nn.BatchNorm2d(512)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        
    def forward(self,x) :
        #TODO:
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.maxPool2d(x)
        x = self.norm64(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.maxPool2d(x)
        x = self.norm128(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.maxPool2d(x)
        x = self.norm256(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.maxPool2d(x)
        x = self.norm512(x)

        x = self.GAP(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.softmax(x)

        return x


class MyDataset(Dataset) :

    def __init__(self,meta_path,root_dir,transform=None) :
        super().__init__()
        self.dir = root_dir
        f = open(meta_path, "r")
        data = json.loads(f.read())
        self.labels = data["annotations"]

    def __len__(self) :
        return len(self.labels)

    def __getitem__(self,idx) :
        img_path = os.path.join(self.dir, self.labels[idx]["file_name"])
        try:            
          image = crop(read_image(img_path), 0, 0, 400, 400).float()
          if image.shape[0] == 4:
              image = image[:3]
          label = int(self.labels[idx]["category"])
          return image, label, self.labels[idx]["file_name"]
        except RuntimeError:
          image = crop(read_image("./train_data/0.jpg"), 0, 0, 400, 400).float()
          label = 8
          return image, label, self.labels[idx]["file_name"]

def train() :
    #TODO: Make your own training code
    data_dir = "./train_data"
    meta_path = "./answer.json"
    train_datasets = MyDataset(meta_path, data_dir, None)
    train_data, valid_data = torch.utils.data.random_split(train_datasets, [35000, 5000])
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True, drop_last = True)

    vgg11 = MyModel(3, 80).to(device)
    epochs = 40
    criterion = nn.MSELoss(size_average=False)
    optimizerFast = optim.Adam(vgg11.parameters(), lr=1e-3, weight_decay=0.01)
    optimizerMed = optim.Adam(vgg11.parameters(), lr=1e-4, weight_decay = 0.01)
    optimizerSlow = optim.Adam(vgg11.parameters(), lr=1e-5, weight_decay = 0.01)
    optimizerVSlow = optim.Adam(vgg11.parameters(), lr=1e-6, weight_decay = 0.01)

    for epoch in range(epochs):
        for idx, data in enumerate(train_dataloader):
            X = data[0].to(device)
            Y = data[1].to(device)
            Y = F.one_hot(Y, num_classes = 80).float()
            y = vgg11(X)
            loss = criterion(y, Y)
            if epoch < 10:
              optimizer = optimizerFast
            elif epoch < 20:
              optimizer = optimizerMed
            elif epoch < 30:
              optimizer = optimizerSlow
            else:
              optimizer = optimizerVSlow
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # You SHOULD save your model by
    # You SHOULD not modify the save path
    torch.save(vgg11.state_dict(), "./checkpoint.pth") 
    return

def get_model(model_name, checkpoint_path):
    model = model_name()
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def test():
    
    model_name = MyModel
    checkpoint_path = './model.pth' 
    mode = 'test' 
    data_dir = "./test_data"
    meta_path = "./answer.json"
    model = get_model(model_name,checkpoint_path)

    data_transforms = {
        'train' : None , 
        'test': None
    }

    # Create training and validation datasets
    test_datasets = MyDataset(meta_path, data_dir, data_transforms['mode'])

    # Create training and validation dataloaders
    batch_size=32
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Set model as evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Inference
    result = []
    for images, _, filename in tqdm(test_dataloader):
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(num_image):
            result.append({
                'filename': filename[i],
                'class': preds[i].item()
            })

    result = sorted(result,key=lambda x : int(x['filename'].split('.')[0]))
    
    # Save to csv
    with open('./result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','class'])
        for res in result:
            writer.writerow([res['filename'], res['class']])


def main() :
    train()
    test()
    pass


if __name__ == '__main__':
    main()