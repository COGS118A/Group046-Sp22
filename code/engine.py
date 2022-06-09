import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from tqdm import tqdm
from data import write_to_file

def prepare_model(device, args=None):
    # load model, criterion, optimizer, and learning rate scheduler
    model = get_model(args)
    model = model.to(device)
    
    if args['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    if args['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    if args['lr_scheduling'] == False:
        lr_scheduler = None
    else:
        if args['lr_scheduler'] == 'steplr':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    return model, criterion, optimizer, lr_scheduler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, args=None):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    result_path = args['result']

    best_val_loss = 65535

    print(args)

    for epoch in range(args['epoch']):
        print(f'Epoch {epoch}...')
        train_dataloader, val_dataloader, test_dataloader = dataloaders

        # Train the model
        model.train()
        running_train_loss = 0.0
        for inputs, targets in tqdm(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        train_loss.append(running_train_loss / len(train_dataloader))

        # Start evaluating the model
        model.eval()

        # Calculate the accuracy on train set
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, prediction = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (prediction == targets).sum().item()
            train_acc.append(correct / total * 100)

        # Calculate the loss and accuracy on validation set
        running_val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                # Loss
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                # Accuracy
                _, prediction = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (prediction == targets).sum().item()

            val_loss.append(running_val_loss / len(val_dataloader))
            val_acc.append(correct / total * 100)
        
        print("Loss on train set:", train_loss[-1])
        print(f"Accuracy on train set: {train_acc[-1]}%")
        print("Loss on validation set:", val_loss[-1])
        print(f"Accuracy on validation set: {val_acc[-1]}%")

        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            torch.save(model.state_dict(), f'./{result_path}.pt')
        
        # Adapt learning rate if scheduler is initialized
        if scheduler is not None:
            scheduler.step()
    
    print('Finish training.')

    model.load_state_dict(torch.load(f'./{result_path}.pt'))

    # Calculate the loss and accuracy on test set
    running_test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            # Loss
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item()

            # Accuracy
            _, prediction = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (prediction == targets).sum().item()

        test_loss = running_test_loss / len(test_dataloader)
        test_acc = correct / total * 100

    print("Loss on test set: ", test_loss)
    print(f"Accuracy on test set: {test_acc}%")

    # Save the results for further visualization
    data = {'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc,
            'test_loss': test_loss, 'test_acc': test_acc}
    write_to_file(f'./{result_path}.pkl', data)

    return model # return the model with weight selected by best performance 
