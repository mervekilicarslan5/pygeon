#main function
import cnns
import data_load
import misc
import train

import parameter_export
import torch
import numpy as np
import os
from torch import nn
from torch import optim
#from deepreduce_models.resnet import *

def save_model(model,filepath):
    parameter_export.save_weights_compatible_with_cpp(model, filepath+'.bin')

def export_pth_model(model,filepath):
    torch.save(model.state_dict(), filepath+'.pth')

def train_model(model,dataset_name,num_epochs,lr):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train.train_and_evaluate(model, train_loader, test_loader, num_epochs)
    

def load_model(model,filepath):
    model.load_state_dict(torch.load(filepath+'.pth', map_location=torch.device('cpu')))

def load_checkpoint(model,filepath):
    checkpoint = torch.load(filepath+'.pth.tar', map_location=torch.device('cpu'))
    # Assuming the model is wrapped in DataParallel and saved in 'snet' key
    model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['snet'].items()})

def load_model_quant(quantized_model, model):
    """ Loads in the weights into an object meant for quantization """
    state_dict = model.state_dict()
    model = model.to('cpu')
    quantized_model.load_state_dict(state_dict)

def test_model(model,dataset_name):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name)
    train.evaluate(model, test_loader)

def export_test_dataset(dataset_name):
    train_set, test_set,num_classes = data_load.load_dataset(dataset_name)
    data_load.export_dataset(test_set,'./data/'+dataset_name+'_test_images.bin','./data/'+dataset_name+'_test_labels.bin')

def main():
    
    #train the nomrla lenet
    model = cnns.LeNet(quantization=False) # replace with Qunatized LeNet
    dataset_name = 'MNIST'
    modelpath = './models/lenet5_mnist'
    num_epochs = 8
    lr = 0.01
    train_model(model,dataset_name,num_epochs,lr) # replace with qunatized training function
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name)
    #quantize the model
    lenet_quant = cnns.LeNet(quantization=True)
    torch.backends.quantized.engine = 'qnnpack'

    # Instantiate the LeNet model
    load_model_quant(lenet_quant, model)
    #lenet_quant.load_state_dict(lenet.state_dict())
    # After the model has been converted to a quantized version
    total_params = sum(p.numel() for p in lenet_quant.parameters())
    lenet_quant.eval()
    # Fuse Conv, bn and relu
    #net.fuse_model()


    print(f'Number of parameters in the quantized model: {total_params}')
    # Specify quantization configuration
    lenet_quant.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # Prepare the model for static quantization
    torch.quantization.prepare(lenet_quant, inplace=True)

    # Calibrate the model and collect statistics
    for inputs, _ in train_loader:
        lenet_quant(inputs)

    # Convert to a quantized model
    torch.quantization.convert(lenet_quant, inplace=True)
    # Print scale and zero-point information


    parameter_export.save_quantized_weights_compatible_with_cpp(lenet_quant, modelpath+'.bin') 
    parameter_export.save_quantization_params(lenet_quant, modelpath+'quant'+'.bin')
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name) # replace with qunatized dataset loader
    data_load.export_quantized_dataset(lenet_quant, test_loader,'./data/'+dataset_name+'_test_images.bin','./data/'+dataset_name+'_test_labels.bin') # export qunatized dataset



    # model = DRD_C100_230K(num_classes=100)  # or any other model you have
    # model = DRD_C100_115K(num_classes=100)  # or any other model you have
    # model = cnns.ResNet50(num_classes=100)
    # dataset_name = 'CIFAR-100'
    # modelpath = './deepreduce_models/CIFAR100_models/model_DRD_C100_115K'
    # num_epochs = 20
    # lr = 0.001

    # load_model(model,modelpath)
    # load_model(model,'./models/resnet50_cifar100')
    # save_model(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K_mod')
    # load_checkpoint(model,modelpath)
    # export_pth_model(model,modelpath)

    # parameter_export.write_params(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K.bin')
    # print(model)

    # parameter_export.read_params(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K.bin')
    # parameter_export.import_weights_compatible_with_cpp(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K.bin')
    # test_model(model,dataset_name)
    # misc.print_layers_and_params(model)
    # print(model)
    # save_model(model,modelpath)
    # test_model(model,dataset_name)
    # export_test_dataset(dataset_name)


if __name__ == '__main__':
    main()

