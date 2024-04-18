import torch
import torch.nn as nn
from SAM_utils.bypass_bn import enable_running_stats, disable_running_stats
from SAM_utils.sam import SAM
import clip
import time
# import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import optimize
from dataset import DFADDataset
from models.DFAD_model_base import DFADModel
from tqdm import tqdm

import os


checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device('cuda')

def threshplus_tensor(x):
    y = x.clone()
    pros = torch.nn.ReLU()
    z = pros(y)
    return z

def search_func(losses, alpha):
    return lambda x: x + (1.0/alpha)*(threshplus_tensor(losses-x).mean().item())

def searched_lamda_loss(losses, searched_lamda, alpha):
    return searched_lamda + ((1.0/alpha)*torch.mean(threshplus_tensor(losses-searched_lamda))) 

def calculate_L_AUC(P_scores, N_scores, eta, p):
    # Convert scores to column and row vectors respectively
    P_scores = P_scores.unsqueeze(1)  # Make it a column vector
    N_scores = N_scores.unsqueeze(0)  # Make it a row vector

    # Compute the margin matrix in a vectorized form
    margin_matrix = P_scores - N_scores - eta

    # Apply the ReLU-like condition and raise to power p
    loss_matrix = torch.where(margin_matrix < 0, (-margin_matrix) ** p, torch.zeros_like(margin_matrix))

    # Compute the final L_AUC by averaging over all elements
    L_AUC = loss_matrix.mean()

    return L_AUC



def train_epoch(model, optimizer, scheduler, criterion, train_loader,loss_type, gamma):
    model.train()
    total_loss_accumulator = 0
    alpha_cvar = 0.8
    #------------- L_AUC parameter-------------------#
    eta = 0.6 #(0,1]
    p = 2 # >1
    gamma = gamma
    #------------- L_AUC parameter-------------------#

    def calculate_loss(output, labels, loss_type, criterion, compute_auc=False):
    
        loss_ce = criterion(output, labels)
        # Directly return loss_ce for 'erm' loss type
        if loss_type == 'erm':
            return loss_ce.mean()

        # For 'dag' and 'auc' loss types, perform additional computations
        if loss_type in ['dag', 'auc']:
            chi_loss_np = search_func(loss_ce, alpha_cvar)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            loss = searched_lamda_loss(loss_ce, cutpt, alpha_cvar)
            
            # If compute_auc is True and loss_type is 'auc', compute the AUC component
            if compute_auc and loss_type == 'auc':
                positive_scores = output[labels == 1]
                negative_scores = output[labels == 0]
                loss_auc = calculate_L_AUC(positive_scores, negative_scores, eta, p)
                loss = gamma * loss + (1 - gamma) * loss_auc

        return loss
    
    for inputs, text_inputs, labels in tqdm(train_loader):
        inputs, text_inputs, labels = inputs.to(device), text_inputs.to(device), labels.to(device)
        
        enable_running_stats(model)
        output = model(inputs, text_inputs).squeeze()  # Assuming model accepts both image and text inputs
        total_loss = calculate_loss(output, labels, loss_type, criterion, compute_auc=(loss_type == 'auc'))  
        total_loss.backward()
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model) 
        output = model(inputs, text_inputs).squeeze()
        total_loss = calculate_loss(output, labels, loss_type, criterion, compute_auc=(loss_type == 'auc'))
        total_loss.backward()
        optimizer.second_step(zero_grad=True)

        total_loss_accumulator += total_loss.item()


    scheduler.step()
    return total_loss_accumulator / len(train_loader)  # Return average loss

def evaluate(model, criterion, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, text_inputs, labels in val_loader:  # Adjusted to unpack three items
            inputs, text_inputs, labels = inputs.to(device), text_inputs.to(device), labels.to(device)
            output = model(inputs, text_inputs).squeeze()  # Ensure model accepts both inputs
            probabilities = torch.sigmoid(output)
            predicted = probabilities > 0.5
            total_correct += (predicted.float() == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(probabilities.cpu().numpy())

    accuracy = total_correct / total_samples
    auc_score = roc_auc_score(all_labels, all_predictions)
    return accuracy, auc_score

def model_trainer(loss_type, alpha, batch_size=64, num_epochs=32):
    # Move model to GPU
    print(alpha)
    model = DFADModel()
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7]).cuda()
    train_dataset = DFADDataset('train')
    val_dataset = DFADDataset('val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=32,shuffle=False)


    # Prepare data loaders
    if loss_type == 'erm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    
    # Initialize optimizer and scheduler
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    # Initialize the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=num_epochs / 4, eta_min=1e-5)  # eta_min is the minimum lr
    checkpoint_dir = f'checkpoints_{loss_type}_alpha_{alpha}_new'
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_file_path = os.path.join(checkpoint_dir, 'performance_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        f.write('Epoch,Train Loss,Validation Accuracy,Validation AUC\n')


    # trian and evaluate
    for epoch in range(num_epochs):
        print(str(epoch).zfill(4))
        train_loss = train_epoch(model, optimizer, scheduler, criterion,train_loader,loss_type, alpha)

        # val_loss, accuracy, auc= evaluate(model, criterion, val_loader)
        accuracy, auc= evaluate(model, criterion, val_loader)


        # print(f'Validation Loss: {val_loss:.6f}')
        print(f'train loss: {train_loss:.6f}')
        print(f'Validation Acc: {accuracy:.6f}')
        print(f'Validation AUC: {auc:.6f}')
        print()
        # print(f'Train Loss: {train_loss:.6f}, Validation Accuracy: {accuracy:.6f}, Validation AUC: {auc:.6f}')
        with open(metrics_file_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{accuracy},{auc}\n')

        # Saving model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pt')
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':

    model_trainer(loss_type='auc', gamma=0.5,  batch_size=2048, num_epochs=32)
