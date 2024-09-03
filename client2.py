import base64
import binascii
import json
import zlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
from io import BytesIO
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(12),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(12 * 12, 8)
        # self.fc2 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        x = x.view(-1, 12 * 12)
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x
    
def parameters_to_string(model):
    state_dict = model.state_dict()
    serializable_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, list) or isinstance(value, float):
            # Round floating-point numbers to 7 decimal places
            value = round(value, 7)
        elif isinstance(value, torch.Tensor):
            # Convert PyTorch tensors to lists and round floating-point numbers
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
    
# def parameters_to_string(model):
#     # Serialize model parameters to a dictionary
#     parameters = model.state_dict()
#     # print(parameters)
#     #count number of parameters
#     # for name, param in model.named_parameters():
#     #     print(name, param.shape)
#     #     print(len(param))
#     parameters = ([param.detach().numpy().tolist() for name, param in model_01.named_parameters() if name.endswith(('weight', 'bias'))])
#     model_params_json = json.dumps(parameters)
#     compressed_data = zlib.compress(model_params_json.encode('utf-8'), level=zlib.Z_BEST_COMPRESSION)
#     base64_encoded = base64.b64encode(compressed_data).decode('utf-8')
#     # padded_encoded = base64_encoded.ljust(20000)
#     hex_encoded = binascii.hexlify(base64_encoded.encode('utf-8')).decode('utf-8')
#     return hex_encoded

# def string_to_parameters(hex_encoded):
#     # Convert the hex-encoded string to base64
#     base64_encoded = binascii.unhexlify(hex_encoded.encode('utf-8')).decode('utf-8')
#     compressed_data = base64.b64decode(base64_encoded.encode('utf-8'))
#     model_params_json = zlib.decompress(compressed_data).decode('utf-8')
#     parameters_as_lists = json.loads(model_params_json)
#     parameters = [torch.tensor(param_list, dtype=torch.float32) for param_list in parameters_as_lists]
#     return parameters

def runner(trainloader, testloader, epoch, model):
    # model = MLP().to(device)
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

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_accuracy = 100. * correct / len(testloader.dataset)
    # print(f'Test Accuracy: {test_accuracy:.2f}%')
    return train_accuracy, test_accuracy, model

train_dataset_23 = datasets.MNIST('data', train=True, download=True, transform=transform)
indices = torch.where((train_dataset_23.targets > 1) & (train_dataset_23.targets < 4))
train_dataset_23.data = train_dataset_23.data[indices]
train_dataset_23.targets = train_dataset_23.targets[indices]

test_dataset_23 = datasets.MNIST('data', train=False, download=True, transform=transform)
indices = torch.where((test_dataset_23.targets > 1) & (test_dataset_23.targets < 4))
test_dataset_23.data = train_dataset_23.data[indices]
test_dataset_23.targets = train_dataset_23.targets[indices]

# Set up the data loaders
batch_size = 128
train_loader_23 = torch.utils.data.DataLoader(train_dataset_23, batch_size=batch_size, shuffle=True)
test_loader_23 = torch.utils.data.DataLoader(test_dataset_23, batch_size=batch_size, shuffle=False)

# checking is there any pervious trained model
file_path = 'D:\\Arjun Workspace\\MNIST_Model_Training\\client2.txt'
if os.path.getsize(file_path) == 0:
    model = MLP().to(device)
else:
    with open(file_path, 'r') as f:
        parameter_string = f.read()
        parameters = string_to_parameters(parameter_string)
        model = MLP().to(device)  # Instantiate your model
        state_dict = model.state_dict() 
        state_dict.update({key: value for key, value in zip(state_dict, parameters)})
        model.load_state_dict(state_dict) 
    
train_accuracy, test_accuracy, model_01 = runner(train_loader_23, test_loader_23, 5, model)
with open('D:\\Arjun Workspace\\MNIST_Model_Training\\accuracy2.txt', 'a') as file:
    # print(train_accuracy)
    file.write(str(test_accuracy)+"\n")
encoded_parameters_02 = parameters_to_string(model_01)
print(encoded_parameters_02)

with open('D:\\Arjun Workspace\\MNIST_Model_Training\\client2.txt', 'w') as file:
    file.write(encoded_parameters_02)
# print(len(encoded_parameters_02))
# decode_paramerters_02 = string_to_parameters(encoded_parameters_02)
# print(decode_paramerters_02)


