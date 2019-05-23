import os
import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import BachDataset
from model import BachNet


def train(config):
    date = "{0:%m-%d %H-%M}".format(datetime.datetime.now())
    idString = date + "-lr" + str(config['learning_rate']) + "-g" + str(config['learning_gamma']) +"-ls" + str(config['learning_step']) + "-hs" + str(config['hidden_size']) + "-nh" + str(config['number_hidden']) + "-fs" + str(config['frame_size']) + "-do" + str(config['dropout']) + '-'
    checkpoint_dir = os.path.join('.', 'checkpoints', idString)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f'runs/{idString}')

    data_path = 'data/'


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    params = {'batch_size': 50, 'shuffle': True}

    # Datasets
    dataloaders = {
        'train':
            DataLoader(
                BachDataset(
                    os.path.join(data_path, 'train'), config['frame_size']
                ), **params
            ),
        'valid':
            DataLoader(
                BachDataset(
                    os.path.join(data_path, 'valid'), config['frame_size']
                ), **params
            )
    }

    model = BachNet(
        input_dims=279,
        hidden_dims=config['hidden_size'],
        output_dims=62,
        num_hidden_layers=config['number_hidden'],
        dropout=config['dropout']
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(
        optimizer,
        step_size=config['learning_step'],
        gamma=config['learning_gamma']
    )

    for epoch in tqdm(range(config['number_epochs']), unit='epoch'):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(dataloaders[phase]):
                batch, labels = batch_sample
                batch, labels = batch.to(device), labels.to(device)
                batch = batch.permute(2, 0, 1)
                labels = labels.permute(2, 0, 1)

                with torch.set_grad_enabled(phase == 'train'):
                    h = model.init_hidden(len(batch[0]))
                    y_pred_B, y_pred_A, y_pred_T, hidden = model(batch, h)

                    current_batchsize = y_pred_A[0, :, 0].shape[0]
                    beforepad = torch.zeros(config['frame_size'], current_batchsize, 8).to(device)          # padding for format
                    afterpad = torch.zeros(config['frame_size'], current_batchsize, 85).to(device)        # padding for format

                    y_pred = torch.cat((beforepad, y_pred_B, y_pred_T, y_pred_A, afterpad), 2)

                    # loss = criterion(y_pred, labels)
                    b_labels = labels[:, :, 8:70]
                    #b_labels = (torch.max(b_labels, 2)[1]).contiguous().view(-1)
                    #y_pred_B = y_pred_B.view(-1, current_batchsize * config['frame_size']).permute(1, 0)

                    a_labels = labels[:, :, 70:132]
                    #a_labels = (torch.max(a_labels, 2)[1]).contiguous().view(-1)
                    #y_pred_A = y_pred_A.view(-1, current_batchsize * config['frame_size']).permute(1, 0)

                    t_labels = labels[:, :, 132:194]
                    #t_labels = (torch.max(t_labels, 2)[1]).contiguous().view(-1)
                    #y_pred_T = y_pred_T.view(-1, current_batchsize * config['frame_size']).permute(1, 0)


                    for ts in range(config['frame_size']):
                        y_pred_B_ts = y_pred_B [ts, :, :]
                        y_pred_A_ts = y_pred_A [ts, :, :]
                        y_pred_T_ts = y_pred_T [ts, :, :]

                        b_labels_ts = b_labels[ts, :, :]
                        b_labels_ts = (torch.max(b_labels_ts, 1)[1])
                        a_labels_ts = a_labels[ts, :, :]
                        a_labels_ts = (torch.max(a_labels_ts, 1)[1])
                        t_labels_ts = t_labels[ts, :, :]
                        t_labels_ts = (torch.max(t_labels_ts, 1)[1])

                        y_pred_B_ts = y_pred_B_ts.squeeze()
                        y_pred_A_ts = y_pred_A_ts.squeeze()
                        y_pred_T_ts = y_pred_T_ts.squeeze()
                        b_labels_ts = b_labels_ts.squeeze()
                        a_labels_ts = a_labels_ts.squeeze()
                        t_labels_ts = t_labels_ts.squeeze()

                        print(y_pred_B_ts.shape)

                        loss_b = criterion(y_pred_B_ts, b_labels_ts)
                        loss_a = criterion(y_pred_A_ts, a_labels_ts)
                        loss_t = criterion(y_pred_T_ts, t_labels_ts)
                        # loss1 = criterion(y_pred.view(-1,current_batchsize * config['frame_size']), labels.contiguous().view(-1,current_batchsize * config['frame_size']))
                        # loss = lossfn(scores.view(-1,batch_size*time_steps), labels.contiguous().view(-1))

                        loss = sum([loss_b, loss_a, loss_t])

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward(retain_graph=True)
                            optimizer.step()

                    step = int(
                        (float(epoch) + (batch_idx / len(dataloaders[phase]))) *
                        1000
                    )
                    writer.add_scalars('loss', {phase: loss.item()}, step)

        if config['save_checkpoints']:
            tempName = os.path.join(checkpoint_dir, idString + str(epoch) + ".pt")
            torch.save({'state': model.state_dict(), 'config': config}, tempName)


if __name__ == '__main__':
    from random import choice, randint

    config = {
        'learning_rate': 0.0001,
        'learning_gamma': 0.9,
        'learning_step': 3,
        'hidden_size': 45,
        'number_hidden': 2,
        'frame_size': 16,
        'dropout': 0.5,
        'number_epochs': 100,
        'save_checkpoints': True,
    }

    train(config)
