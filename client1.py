import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(12 * 12, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = x.view(-1, 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def parameters_to_string(model):
    state_dict = model.state_dict()
    serializable_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, list) or isinstance(value, float):
            value = round(value, 7)
        elif isinstance(value, torch.Tensor):
            value = value.tolist()
            for i in range(len(value)):
                if isinstance(value[i], list):
                    value[i] = [round(x, 7) for x in value[i]]
        serializable_state_dict[key] = value
    state_dict_json = json.dumps(serializable_state_dict, indent=4)
    return state_dict_json

def string_to_parameters(json_string):
    serializable_state_dict = json.loads(json_string)
    state_dict = {}
    for key, value in serializable_state_dict.items():
        if isinstance(value, list):
            # If the value is a list, convert it back to a torch.Tensor
            value = torch.tensor(value)
        state_dict[key] = value
    return state_dict

def runner(trainloader, epoch, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = epoch
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

        train_accuracy = 100.0 * total_correct / total_samples
        # print(f'Training Accuracy for Epoch {epoch + 1}: {train_accuracy:.2f}%')

    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in testloader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += criterion(output, target).item()
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(testloader.dataset)
    # test_accuracy = 100. * correct / len(testloader.dataset)
    # print(f'Test Accuracy: {test_accuracy:.2f}%')
    return train_accuracy, model

data_transform = transforms.Compose([
    transforms.Resize((12, 12)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Define the data loaders for the two folders
dataset_01 = ImageFolder('./mnist_01', transform=data_transform)

batch_size = 64
train_loader_01 = DataLoader(dataset_01, batch_size=batch_size, shuffle=True)

# import re
# def remove_null_characters(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         content = infile.read()
#     cleaned_content = content.replace('\x00', '')
#     content = content[:40000]
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         outfile.write(cleaned_content)
        
# def clean_text(text):
#     text = re.sub('[^0-9a-fA-F]', '0', text)  # Use a string pattern and replacement
#     return text

# remove_null_characters('D:\\Arjun Workspace\\MNIST_Model_Training\\Client12Server.txt', 'D:\\Arjun Workspace\\MNIST_Model_Training\\Client12Server.txt')
file_path = "D:\\Arjun Workspace\\MNIST_Model_Training\\Client12Server.txt"
if os.path.getsize(file_path) == 0:
    model = MLP().to(device)
else:
    with open(file_path, 'r') as f:
        parameter_string = f.read()
        # parameter_string = clean_text(parameter_string)
        parameters = string_to_parameters(parameter_string)
        model = MLP().to(device)  # Instantiate your model
        state_dict = model.state_dict() 
        state_dict.update({key: value for key, value in zip(state_dict, parameters)})
        model.load_state_dict(state_dict) 

train_accuracy, model_01 = runner(train_loader_01, 5, model)
decoded_parameters_01 = parameters_to_string(model_01)
print(len(decoded_parameters_01))
decoded_parameters_01 = decoded_parameters_01.replace('\n', '')
decoded_parameters_01 = decoded_parameters_01.replace(' ', '')
print(len(decoded_parameters_01))
print((decoded_parameters_01))
# print(len(decoded_parameters_01))
with open('D:\\Arjun Workspace\\MNIST_Model_Training\\Client12Server.txt', 'w') as file:
    file.write(decoded_parameters_01)

