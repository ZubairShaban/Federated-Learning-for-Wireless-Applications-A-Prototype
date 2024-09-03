import binascii
import zlib
import torch
from io import BytesIO
import cleaner as cl
import re
import json
import base64

def clean_hex_string(hex_string):
    cleaned_hex = re.sub(rb'[^\x20-\x7E]', b'', hex_string)
    cleaned_hex = re.sub(rb'[^0-9a-fA-F]', b'0', cleaned_hex)
    return cleaned_hex


def clean_function(input_file, output_file):
    concatenated_output = b''
    # Open the input file in binary mode for reading
    with open(input_file, 'rb') as input_file:
        for line in input_file:
            # Clean each line from the input file
            cleaned_hex = clean_hex_string(line)
            if cleaned_hex:
                concatenated_output += cleaned_hex
    with open(output_file, 'wb') as output_file:
        output_file.write(concatenated_output)
    concatenated_output_str = concatenated_output.decode('utf-8')
    return(concatenated_output_str)

def parameters_to_string(model):
    # Serialize model parameters to a dictionary
    parameters = model.state_dict()
    print(parameters)
    parameters = ([param.detach().numpy().tolist() for name, param in model_01.named_parameters() if name.endswith(('weight', 'bias'))])
    model_params_json = json.dumps(parameters)
    compressed_data = zlib.compress(model_params_json.encode('utf-8'), level=zlib.Z_BEST_COMPRESSION)
    base64_encoded = base64.b64encode(compressed_data).decode('utf-8')
    # padded_encoded = base64_encoded.ljust(20000)
    hex_encoded = binascii.hexlify(base64_encoded.encode('utf-8')).decode('utf-8')
    return hex_encoded

def string_to_parameters(hex_encoded):
    # Convert the hex-encoded string to base64
    base64_encoded = binascii.unhexlify(hex_encoded.encode('utf-8')).decode('utf-8')
    compressed_data = base64.b64decode(base64_encoded.encode('utf-8'))
    model_params_json = zlib.decompress(compressed_data).decode('utf-8')
    parameters_as_lists = json.loads(model_params_json)
    parameters = [torch.tensor(param_list, dtype=torch.float32) for param_list in parameters_as_lists]
    return parameters
    
def is_hex_string(input_string):
    # Use regular expression to check if the string consists of hexadecimal characters
    return re.match(r"^[0-9a-fA-F]+$", input_string) is not None

def function():
    data_1 = clean_function('D:\\Arjun Workspace\\MNIST_Model_Training\\client1.txt', 'D:\\Arjun Workspace\\MNIST_Model_Training\\client1.txt')
    data_2 = clean_function('D:\\Arjun Workspace\\MNIST_Model_Training\\client2.txt', 'D:\\Arjun Workspace\\MNIST_Model_Training\\client2.txt')
    print(is_hex_string(data_1))
    print(is_hex_string(data_2))
    print(len(data_1))
    print(len(data_2))
    param_01 = string_to_parameters(data_1) 
    param_02 = string_to_parameters(data_2)
    # Federated averaging
    avg_param = []
    for p1, p2 in zip(param_01, param_02):
        avg = (p1 + p2) / 2
        avg_param.append(avg)
    with open('D:\\Arjun Workspace\\MNIST_Model_Training\\average.txt', 'w') as file:
        file.write(parameters_to_string(avg_param))
    return parameters_to_string(avg_param)

print(function())
