import json
import logging
import os
import sys
from PIL import ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time # for measuring time for testing, remove for students
from PIL import ImageFile
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion):
    logger.info("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0

    for inputs, labels in test_loader:

        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")

def train(model,epochs, train_loader, validation_loader, criterion, optimizer):
    #epochs=2
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%) Time: {}".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                            time.asctime() # for measuring time for testing, remove for students and in the formatting
                        )
                    )

                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model

def create_model():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 128),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, 5))
    return model

#batch_size=10

def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get train data loader")
    logger.info("Get validation data loader")
    logger.info("Get test data loader")
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



 #lr=0.001


def main(args):
    '''
     Initialize a model by calling the net function
    '''
    model=create_model()

    '''
    Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    model=train(model, args.epochs, train_loader, validation_loader, criterion, optimizer)

    '''
    Test the model to see its accuracy
    '''

    test(model, test_loader, criterion)


    '''
    Save the trained model
    '''
    logger.info("saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--epochs", type=int, default=2, metavar="E", help="learning rate (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.01)"
    )
    # Container environment


    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()

    main(args)
