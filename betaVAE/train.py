# -*- coding: utf-8 -*-
# /usr/bin/env python3

import numpy as np
import pandas as pd
from torchsummary import summary

from vae import *
from deep_folding.utils.pytorchtools import EarlyStopping


def train_vae(config, trainloader, valloader, root_dir=None):
    """ Trains beta-VAE for a given hyperparameter configuration
    Args:
        config: instance of class Config
        trainloader: torch loader of training data
        valloader: torch loader of validation data
        root_dir: str, directory where to save model

    Returns:
        vae: trained model
        final_loss_val
    """
    torch.manual_seed(0)
    lr = config.lr
    vae = VAE(config.in_shape, config.n, depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)
    summary(vae, config.in_shape)

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    nb_epoch = config.nb_epoch
    early_stopping = EarlyStopping(patience=12, verbose=True, root_dir=root_dir)

    list_loss_train, list_loss_val = [], []

    # arrays enabling to see model reconstructions
    id_arr, phase_arr, input_arr, output_arr = [], [], [], []

    for epoch in range(config.nb_epoch):
        running_loss = 0.0
        epoch_steps = 0
        for inputs, path in trainloader:
            optimizer.zero_grad()

            inputs = Variable(inputs).to(device, dtype=torch.float32)
            target = torch.squeeze(inputs, dim=1).long()
            output, z, logvar = vae(inputs)
            recon_loss, kl, loss = vae_loss(output, target, z,
                                    logvar, criterion,
                                    kl_weight=config.kl)
            output = torch.argmax(output, dim=1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
        print("[%d] loss: %.3f" % (epoch + 1,
                                        running_loss / epoch_steps))
        list_loss_train.append(running_loss / epoch_steps)
        running_loss = 0.0

        """ Saving of reconstructions for visualization in Anatomist software """
        if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('train')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        vae.eval()
        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                target = torch.squeeze(inputs, dim=1).long()
                recon_loss_val, kl_val, loss = vae_loss(output, target,
                                        z, logvar, criterion,
                                        kl_weight=config.kl)
                output = torch.argmax(output, dim=1)

                val_loss += loss.cpu().numpy()
                val_steps += 1
        valid_loss = val_loss / val_steps
        print("[%d] validation loss: %.3f" % (epoch + 1, valid_loss))
        list_loss_val.append(valid_loss)

        early_stopping(valid_loss, vae)

        """ Saving of reconstructions for visualization in Anatomist software """
        if early_stopping.early_stop or epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
            break
    for key, array in {'input': input_arr, 'output' : output_arr,
                           'phase': phase_arr, 'id': id_arr}.items():
        np.save(config.save_dir+key, np.array([array]))

    plot_loss(list_loss_train[1:], config.save_dir+'tot_train_')
    plot_loss(list_loss_val[1:], config.save_dir+'tot_val_')
    final_loss_val = list_loss_val[-1:]

    """Saving of trained model"""
    torch.save((vae.state_dict(), optimizer.state_dict()),
                config.save_dir + 'vae.pt')

    print("Finished Training")
    return vae, final_loss_val
