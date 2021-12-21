import dataset
import torch
from torch import nn
from transformations import train_transform, val_transform
from torch.utils.data import DataLoader
from model import UNet
from tqdm import tqdm
import torch.optim as optim
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
NB_CLASS = 43
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    print("####################################")
    print("The trainer function")
    print("####################################")
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    print("####################################")
    print("Build dataset.MyDataset")
    print("####################################")
    training_data = dataset.MyDataset(
        train=True,
        transform=train_transform()
    )

    val_data = dataset.MyDataset(
        train=False,
        transform=val_transform()
    )

    print("####################################")
    print("Build data loader")
    print("####################################")
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print("####################################")
    print("Init Model, Loss and Optimizer")
    print("####################################")
    model = UNet()
    print("####################################")
    print("Model structure")
    print("####################################")
    print(model)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_dataloader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    print("####################################")
    print("Start training")
    print("####################################")
    for epoch in range(NUM_EPOCHS):
        train_fn(train_dataloader, model, optimizer, loss, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_dataloader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_dataloader, model, folder="saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()