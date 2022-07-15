import torch
from torch.utils.data import DataLoader

import os
import numpy as np
from sklearn.model_selection import train_test_split

from observation.dataset import PCDDataset
from observation.model import ObservationModel


if __name__=="__main__":

    print("Device: " + "cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True

    BATCH_SIZE = 16


    dataset = PCDDataset(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../dataset/"))
    trainset, validset = train_test_split(dataset, train_size=1400, test_size=600, random_state=42)

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )

    validloader = DataLoader(
        dataset=validset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )

    print("Initializing model...")
    model = ObservationModel().to(DEVICE)
    sigmoid = torch.nn.Sigmoid()
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"checkpoint.pt"), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model initialized! Start testing...")

    # Test loop
    model.eval()
    with torch.no_grad():
        train_accuracy = np.array([], dtype=float)
        for (x, y) in trainloader:
            # Feedforward
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            # Acuraccy            
            result = torch.heaviside(torch.add(sigmoid(out), -0.5), torch.tensor([0.0]).to(DEVICE))
            accuracy = torch.div(torch.sum(torch.eq(result, y), dim=-1), float(y.shape[-1]))
            train_accuracy = np.concatenate((train_accuracy, accuracy.to("cpu").numpy()))

        train_acc_mean = np.mean(train_accuracy)        
        print(f"Train accuracy : {train_acc_mean}")


        valid_accuracy = np.array([], dtype=float)
        for (x, y) in validloader:
            # Feedforward
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            
            # Accuracy
            result = torch.heaviside(torch.add(sigmoid(out), -0.5), torch.tensor([0.0]).to(DEVICE))
            accuracy = torch.div(torch.sum(torch.eq(result, y), dim=-1), float(y.shape[-1]))
            valid_accuracy = np.concatenate((valid_accuracy, accuracy.to("cpu").numpy()))

        valid_acc_mean = np.mean(valid_accuracy)        
        print(f"Valid accuracy : {valid_acc_mean}")
        
