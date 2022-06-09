import torch
import argparse
import os, sys, json
from datetime import datetime
from data import get_dataloaders
from engine import *

parser = argparse.ArgumentParser()

parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')    

# Model Related
parser.add_argument('--model', default='baseline', type=str,
                    help='Model being used')

# Data Related
parser.add_argument('--bz', default=4, type=int,
                    help='batch size')
parser.add_argument('--shuffle_data', default=1, type=int,
                    help='Shuffle the data')


# Other Choices & hyperparameters
parser.add_argument('--epoch', default=60, type=int,
                    help='number of epochs')
    # for loss
parser.add_argument('--criterion', default='cross_entropy', type=str,
                    help='which loss function to use')
    # for optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    help='which optimizer to use')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
# for scheduler
parser.add_argument('--lr_scheduling', default=0, type=int,
                    help='Enable learning rate scheduling')
parser.add_argument('--lr_scheduler', default='steplr', type=str,
                    help='learning rate scheduler') 
parser.add_argument('--step_size', default=30, type=int,
                    help='Period of learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay')

parser.add_argument('--result', type=str)

args = vars(parser.parse_args())

def main(args):
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else 'cpu')
    dataloaders = get_dataloaders('data/train_resized.hdf5', 'data/study_label.csv', args)
    model, criterion, optimizer, lr_scheduler = prepare_model(device, args)
    
    model = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders, args)

if __name__ == '__main__':
    main(args)
