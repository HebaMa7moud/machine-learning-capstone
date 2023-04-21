#TODO: Import your dependencies.
import json
import os
import sys
from PIL import ImageFile
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
#import torchvision.transforms as transforms
from torchvision import datasets, transforms
import smdebug.pytorch as smd

import argparse


def test(model, test_loader, loss_criterion, hook):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = loss_criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target.data)
        total_loss = test_loss / len(test_loader.dataset)
        total_acc = correct/ len(test_loader.dataset)

        print(f"Testing Loss: {total_loss}")
        print("Test Accuracy:{:.4f}".format(100*total_acc))


def train(model, epochs,train_loader, validation_loader, loss_criterion, optimizer, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=epochs
   
    image_data= {'train':train_loader, 'valid':validation_loader}
    

    for epoch in range(epochs):
        print("Epoch".format(epoch))
        for img_dataset in ['train', 'valid']:
            if img_dataset== 'train':
                print("START TRAINING")
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0
            corrects = 0
            for data, target in image_data[img_dataset]:
                output = model(data)
                loss = loss_criterion(output, target)
                if img_dataset=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                running_loss += loss.item() * data.size(0)
                _, preds = torch.max(output, 1)
                corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss / len(image_data[img_dataset].dataset)
            epoch_acc = corrects / len(image_data[img_dataset].dataset)
            print('{} loss: {:.4f}, acc: {:.4f}'.format(img_dataset, epoch_loss, epoch_acc))

            if img_dataset=='valid':
                if epoch == epochs-1:
                    print('{} loss: {:.4f}, acc: {:.4f}'.format(img_dataset, epoch_loss, epoch_acc))
                
        if epoch ==0:
            break
    return model



def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained =True)
    num_features = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 128),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, 5))

    return model


def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    print("Get train data loader")
    print("Get validation data loader")
    print("Get test data loader")
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    training_dir = os.path.join(data_dir, "train/")
    valid_dir = os.path.join(data_dir, "valid/")
    test_dir = os.path.join(data_dir , "test/")

    train_transform = transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    valid_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(training_dir, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True)

    validation_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    #register hook to save output
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=5)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    model=train(model, args.epochs, train_loader, validation_loader, loss_criterion, optimizer, hook)

    '''
    TODO: Test the model to see its accuracy
    '''

    test(model, test_loader, loss_criterion, hook)


    '''
    TODO: Save the trained model
    '''
    print("saving the model.")
    path = os.path.join(args.model_dir, "debugged_model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--epochs", type=int, default=5, metavar="E", help="learning rate (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    # Container environment


    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()

    main(args)
