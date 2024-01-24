import torch
import numpy as np


def save_quantized_weights_compatible_with_cpp(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []

        # Process each module according to its type
        for module in model.modules():
            if isinstance(module, torch.nn.quantized.Conv2d):
                # Dequantize the weights and biases before saving
                weight = module.weight().cpu().int_repr().numpy().ravel()
                all_params.append(weight)
                print("conv2d weight", len(weight))
                # Handle the bias
                if module.bias() is not None:
                    bias = module.bias().cpu().detach()
                    scale_input = module.scale # scale of the input
                    scale_weight = module.weight().q_scale() # scale of the weights
                    zero_point_bias =  0 #module.weight().zero_point() #zero point for the bias

                    # Quantize the bias
                    quantized_bias = torch.round(bias / (scale_input * scale_weight)) + zero_point_bias
                    print('quant bias conv2d', len(quantized_bias))
                    all_params.append(quantized_bias.numpy().astype(np.int32))

                    #all_params.append(bias)
                zero_point = module.zero_point
                if zero_point is not None:
                    print('zero_point conv2d', zero_point)
                    all_params.append(np.array([zero_point], dtype=np.int32))
            elif isinstance(module, torch.nn.quantized.Linear):
                # Dequantize the weights and biases before saving
                weight = module.weight().int_repr().numpy().ravel()
                all_params.append(weight)
                print("llinear weight", len(weight))
                # LAYER scales and zero points
                # Handle the bias
                if module.bias() is not None:
                    bias = module.bias().cpu().detach()
                    scale_input = module.scale # scale of the input
                    scale_weight = module.weight().q_scale() # scale of the weights
                    zero_point_bias = 0 #module.weight().zero_point()

                    # Quantize the bias
                    quantized_bias = torch.round(bias / (scale_input * scale_weight)) + zero_point_bias
                    all_params.append(quantized_bias.numpy().astype(np.int32))
                    print('linear bias', len(quantized_bias))
                zero_point = module.zero_point
                
                if zero_point is not None:
                    print('linear zeropoint', zero_point)
                    all_params.append(np.array([zero_point], dtype=np.int32))
                
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                all_params.append(module.running_mean.numpy().ravel())
                all_params.append(module.running_var.numpy().ravel())
                if module.weight is not None:
                    all_params.append(module.weight.int_repr().numpy().ravel())
        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.int32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())


def save_quantization_params(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []

        for module in model.modules():
            if isinstance(module, torch.nn.quantized.Linear) or isinstance(module, torch.nn.quantized.Conv2d):
                # Handle the scale and zero-point
                scale = module.scale
                
                if scale is not None:
                    print(module, "scales", scale)
                    all_params.append(np.array([scale], dtype=np.float32))


        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.float32)
        print(all_params_flat)
        # Write the parameters to the file
        f.write(all_params_flat.tobytes())


def save_weights_compatible_with_cpp(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []

        # Process each module according to its type
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, torch.nn.Conv2d):
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                all_params.append(module.running_mean.cpu().numpy().ravel())
                all_params.append(module.running_var.cpu().numpy().ravel())
                all_params.append(module.weight.data.cpu().numpy().ravel())
                all_params.append(module.bias.data.cpu().numpy().ravel())

        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.float32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())


def count_exported_params(model):
    total_params = 0
    for module in model.modules():  # `modules()` will iterate over all modules in the network
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Count weights and biases
            total_params += module.weight.data.nelement()
            if module.bias is not None:
                total_params += module.bias.data.nelement()

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # Count all BatchNorm parameters
            total_params += module.running_mean.nelement()
            total_params += module.running_var.nelement()
            if module.weight is not None:
                total_params += module.weight.data.nelement()
            if module.bias is not None:
                total_params += module.bias.data.nelement()

    print("Total number of parameters: {}".format(total_params))
    return total_params


def write_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'wb') as f:
        # Write the total number of parameters first
        np.array([total_params], dtype=np.int32).tofile(f)

        # Then write the parameters for each module
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Flatten and save weights
                np_weights = module.weight.data.numpy().flatten()
                np_weights.tofile(f)

                # Save biases
                if module.bias is not None:
                    np_bias = module.bias.data.numpy()
                    np_bias.tofile(f)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Save BatchNorm parameters
                np_mu = module.running_mean.numpy()
                np_var = module.running_var.numpy()
                np_gamma = module.weight.data.numpy()
                np_beta = module.bias.data.numpy()

                np_mu.tofile(f)
                np_var.tofile(f)
                np_gamma.tofile(f)
                np_beta.tofile(f)
def write_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'wb') as f:
        # Write the total number of parameters first
        np.array([total_params], dtype=np.int32).tofile(f)
        print(f"Total parameters: {total_params}")

        for module in model.modules():
            print(module)
            if isinstance(module, torch.nn.Linear):
                # Print and write weights, then biases for Linear layers
                print_layer_info("Linear", module.weight.data.nelement(), module.bias.data.nelement() if module.bias is not None else 0)
                module.weight.data.numpy().flatten().tofile(f)
                if module.bias is not None:
                    module.bias.data.numpy().tofile(f)

            elif isinstance(module, torch.nn.Conv2d):
                # Print and write kernel weights, then biases for Conv2d layers
                print_layer_info("Conv2d", module.weight.data.nelement(), module.bias.data.nelement() if module.bias is not None else 0)
                module.weight.data.numpy().flatten().tofile(f)
                if module.bias is not None:
                    module.bias.data.numpy().tofile(f)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Print and write BatchNorm parameters
                gamma_beta_count = module.weight.data.nelement() + module.bias.data.nelement() if module.weight is not None and module.bias is not None else 0
                print_layer_info("BatchNorm", module.running_mean.nelement(), module.running_var.nelement(), gamma_beta_count)
                module.running_mean.numpy().tofile(f)
                module.running_var.numpy().tofile(f)
                if module.weight is not None:
                    module.weight.data.numpy().tofile(f)
                if module.bias is not None:
                    module.bias.data.numpy().tofile(f)

def print_layer_info(layer_type, weight_count, bias_count, additional_count=0):
    total = weight_count + bias_count + additional_count
    print(f"{layer_type} Layer - Weights: {weight_count}, Biases: {bias_count}, Additional: {additional_count}, Total: {total}")

def read_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'rb') as f:
        # Read the total number of parameters first
        num_params = np.fromfile(f, dtype=np.int32, count=1)[0]
        if num_params != total_params:
            raise ValueError("Expected {} parameters, but found {}.".format(total_params, num_params))
        for module in model.modules():
            
            if isinstance(module, torch.nn.Linear):
                # Read and reshape weights, then biases
                read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

            elif isinstance(module, torch.nn.Conv2d):
                # Read and reshape kernel weights, then biases
                read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Read BatchNorm parameters in order: moving mean, variance, gamma, beta
                read_to_tensor(f, module.running_mean)
                read_to_tensor(f, module.running_var)
                if module.weight is not None:
                    read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

def read_to_tensor(file, tensor):
    num_elements = tensor.numel()
    tensor_data = np.fromfile(file, dtype=np.float32, count=num_elements)
    tensor.data.copy_(torch.from_numpy(tensor_data).view_as(tensor))

