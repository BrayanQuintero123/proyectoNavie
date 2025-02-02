import os 

import torch

from torch import nn

import argparse

from torchvision import transforms

import data_setup, engine, model_builder, utils

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the DataLoader")
parser.add_argument("--train_dir", type=str, default="data/seg_train/seg_train", help="Directory containing the training data")
parser.add_argument("--test_dir", type=str, default="data/seg_test/seg_test", help="Directory containing the test data")
parser.add_argument("--hidden_units", type=int, default=64, help="Number of hidden units between layers")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
TRAIN_DIR = args.train_dir
TEST_DIR = args.test_dir
print(f"[INFO] training a model for {NUM_EPOCHS} epochs, ...")

# device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

# transforms

data_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

#create dataloader
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    transform=data_transforms,
    batch_size=BATCH_SIZE
)

# create model
model = model_builder.CNN(input_shape=3, hidden_units=HIDDEN_UNITS, 
                          output_shape=len(class_names)
                          ).to(device)

# create loss and optimzier 

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#start training 

train_loss, train_acc = engine.train_step(model=model,
                        train_dataloader=train_dataloader,
                        loss_fn=loss,
                        optimizer=optimizer,
                        device=device)

#show results of training

print(f"[INFO] train Loss: {train_loss:.4f}, train Accuracy: {train_acc:.4f}")

# save mode 

utils.save_model(model=model,
                target_dir="models",
                model_name="0.first_model.pth")