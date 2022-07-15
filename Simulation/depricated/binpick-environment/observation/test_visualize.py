import torch
from torch.utils.data import DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt
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

    print("Initializing model...")
    model = ObservationModel().to(DEVICE)
    sigmoid = torch.nn.Sigmoid()
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"checkpoint.pt"), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model initialized! Start testing...")

    # Test loop
    model.eval()
    with torch.no_grad():

        for i in range(5):

            x, y = validset[i]
            # Feedforward
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            norm_out = sigmoid(out)

            pcd = x.to("cpu").numpy()
            pcd = np.reshape(pcd, (-1, 3))
            prob = norm_out.to("cpu").numpy()
            binary = torch.heaviside(torch.add(norm_out, -0.5), torch.tensor([0.0]).to(DEVICE)).to("cpu").numpy()
            gt = y.to("cpu").numpy()

            # plot prob
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(131, projection='3d')   
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_xlim([0, 0.5])
            ax1.set_ylim([-0.25, 0.25])
            ax1.set_zlim([-0.25, 0.25])
            
            ax1.set_title("Observation Probability", fontsize=10)
            ax1.view_init(elev=30, azim=150)
            im = ax1.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=prob, vmin=0, vmax=1)


            # plot binary
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_xlim([0, 0.5])
            ax2.set_ylim([-0.25, 0.25])
            ax2.set_zlim([-0.25, 0.25])
            
            ax2.set_title("Binary Classification", fontsize=10)
            ax2.view_init(elev=30, azim=150)
            ax2.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=binary, vmin=0, vmax=1)

            # plot binary
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_xlim([0, 0.5])
            ax3.set_ylim([-0.25, 0.25])
            ax3.set_zlim([-0.25, 0.25])
            
            ax3.set_title("Ground Truth", fontsize=10)
            ax3.view_init(elev=30, azim=150)
            ax3.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=gt, vmin=0, vmax=1)
            
            cbaxes = fig.add_axes([0.02, 0.1, 0.02, 0.8])
            fig.colorbar(im, ax=[ax1, ax2, ax3], cax=cbaxes)



        plt.show()
