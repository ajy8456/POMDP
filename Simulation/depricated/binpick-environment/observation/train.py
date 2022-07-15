import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from observation.dataset import PCDDataset
from observation.model import ObservationModel

class LearningRateScheduler(torch.optim.lr_scheduler.LambdaLR):
    '''
    Learning rate scheduler for training.
    - Warmups for 5 epochs
    - x 0.1 after 100 epochs
    - x 0.01 after 200 epochs
    '''
    def __init__(self, optimizer, last_epoch=-1, warmup_steps=5):

        def lr_lambda(step):
            if step < warmup_steps:
                factor = 0.1 + float(step)*0.9/float(warmup_steps)
            elif step < 800:
                factor = 1.
            elif step < 1000:
                factor = 0.1
            else:
                factor = 0.01

            return factor

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch, verbose=True)


if __name__=="__main__":

    print("Device: " + ("cuda" if torch.cuda.is_available() else "cpu"))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True

    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 1000
    WEIGHT_DECAY = 0.0005

    #tensorboard --logdir runs
    writer = SummaryWriter()

    
    dataset = PCDDataset(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../dataset/"))
    trainset, validset = train_test_split(dataset, train_size=1400, test_size=600, random_state=42)

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    validloader = DataLoader(
        dataset=validset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )


    print("Initializing model... Device: " + DEVICE)
    model = ObservationModel().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = LearningRateScheduler(optimizer)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()
    print("Model initialized! Start training...")

    # ============
    # Epochs
    # ============
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")

        # Train loop
        model.train()
        train_losses = np.array([], dtype=float)
        train_accuracy = np.array([], dtype=float)

        loop = tqdm(trainloader)
        for (x, y) in loop:
            # Feedforward
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            # Get loss
            loss = loss_fn(out, y)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update progress
            loop.set_postfix(train_losses=loss.item())
            train_losses = np.concatenate((train_losses, np.full((x.shape[0]), loss.item())))
            # Acuraccy            
            result = torch.heaviside(torch.add(sigmoid(out), -0.5), torch.tensor([0.0]).to(DEVICE))
            accuracy = torch.div(torch.sum(torch.eq(result, y), dim=-1), float(y.shape[-1]))
            train_accuracy = np.concatenate((train_accuracy, accuracy.to("cpu").numpy()))


        # Print stat


        # Validation Loop
        model.eval()
        valid_losses = np.array([], dtype=float)
        valid_accuracy = np.array([], dtype=float)

        with torch.no_grad():
            loop = tqdm(validloader)
            for (x, y) in loop:
                # Feedforward
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                # Get loss
                loss = loss_fn(out, y)
                # Update progress
                loop.set_postfix(valid_losses=loss.item())
                valid_losses = np.concatenate((valid_losses, np.full((x.shape[0]), loss.item())))
                # Accuracy
                result = torch.heaviside(torch.add(sigmoid(out), -0.5), torch.tensor([0.0]).to(DEVICE))
                accuracy = torch.div(torch.sum(torch.eq(result, y), dim=-1), float(y.shape[-1]))
                valid_accuracy = np.concatenate((valid_accuracy, accuracy.to("cpu").numpy()))

            model.train()
        
        # Print stat
        train_mean_loss = np.mean(train_losses)
        train_acc_mean = np.mean(train_accuracy)
        print(f"Mean train loss:   {train_mean_loss:.8f}, Mean train acc:   {train_acc_mean:.8f}, Samples:   {len(trainset)}")

        valid_mean_loss = np.mean(valid_losses)
        valid_acc_mean = np.mean(valid_accuracy)
        print(f"Mean valid loss:   {valid_mean_loss:.8f}, Mean valid acc:   {valid_acc_mean:.8f}, Samples:   {len(validset)}")

        writer.add_scalars("Loss/train,val", {'train': train_mean_loss, 'val': valid_mean_loss}, epoch)
        writer.add_scalars("Acc/train,val", {'train': train_acc_mean, 'val': valid_acc_mean}, epoch)

        scheduler.step()

        writer.flush()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_mean_loss': train_mean_loss,
            'valid_mean_loss': valid_mean_loss
            }, 
            os.path.join(os.path.dirname(os.path.realpath(__file__)),"checkpoint.pt"))


    writer.close()