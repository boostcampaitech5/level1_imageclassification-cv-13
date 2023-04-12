"""
train.py
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

import utils
import data_utils
import models


def train(model, dataloader, criterion, optimizer, device, epoch):
    loss = utils.AverageMeter()
    acc = utils.AverageMeter()

    model.train()

    for inputs, targets in tqdm(dataloader, leave=False, total=100, ncols=80, desc=f'Epoch {epoch:04d} Train'):
        inputs, targets = inputs.to(device), targets.to(device)
        b = targets.size(0)  # batch size
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss_ = criterion(outputs, targets)
        loss_.backward()
        optimizer.step()
        loss.update(loss_.item(), b)

        preds = outputs.argmax(dim=1)
        acc.update(torch.sum(preds==targets).item() / b, b)

    return OrderedDict([('loss', loss.avg), ('acc', acc.avg)])


def eval(model, dataloader, criterion, device, epoch):
    loss = utils.AverageMeter()
    acc = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, leave=False, total=100, ncols=80, desc=f'Epoch {epoch:04d} Valid'):
            inputs, targets = inputs.to(device), targets.to(device)
            b = targets.size(0)  # batch size
            
            outputs = model(inputs)
            
            loss_ = criterion(outputs, targets)
            loss.update(loss_.item(), b)

            preds = outputs.argmax(dim=1)
            acc.update(torch.sum(preds==targets).item() / b, b)

    return OrderedDict([('loss', loss.avg), ('acc', acc.avg)])



def main(args):
    _logger = utils.get_logger(args.exp_name)
    utils.log_configs(args)

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info(f'Device: {DEVICE}')

    utils.seed_everything(args.seed)

    image_files = data_utils.generate_file_list(args.datadir)
    dataset = data_utils.MaskDataset(image_files, args.target, group_age=False)
    train_dataloader, valid_dataloader = data_utils.get_dataloader(dataset,
                                                                   args.batch_size,
                                                                   val_split=args.val_split,
                                                                   shuffle=args.shuffle_off,
                                                                   drop_last=True,
                                                                   )

    model_dict = models.init_model_dict()

    if args.model in model_dict.keys():
        model = getattr(models, model_dict[args.model])(args.n_class)
    else:
       raise ValueError(f"'{args.model}' not implemented!")

    model.to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.optimizer](
        model.parameters(), lr=args.lr
    )

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLr(optimizer, T_max=args.epochs)
    else:
        scheduler = None


    loss_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='Loss', i=1)
    acc_curve = utils.CurvePlotter(title=f'{args.exp_name}', xlabel='Epoch', ylabel='Accuracy', i=2)

    epoch_msg_header = f"{'Epoch':^10} {'Train Loss':^16} {'Train Acc':^15} {'Valid Loss':^16} {'Valid Acc':^15}"
    _logger.info(epoch_msg_header)
    epoch_msg_header = '\n' + '=' * 75 + '\n' + epoch_msg_header + '\n' + '=' * 75
    print(epoch_msg_header)

    for epoch in range(1, args.epochs+1):
        train_metrics = train(model, train_dataloader, criterion, optimizer, DEVICE, epoch)
        valid_metrics = eval(model, valid_dataloader, criterion, DEVICE, epoch)
    
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([(f'train_{k}', v) for k, v in train_metrics.items()])
        metrics.update([(f'valid_{k}', v) for k, v in valid_metrics.items()])

        # epoch_msg = f'Epoch {epoch:04d} '+ ', '.join([f'{k}: {v:.6f}' for k, v in metrics.items()])
        epoch_msg = f"""{f'{epoch:04d}':^10} {f"{metrics['train_loss']:.6f}":^16} {f"{metrics['train_acc']:.4f}":^15} {f"{metrics['valid_loss']:.6f}":^16} {f"{metrics['valid_acc']:.4f}":^15}\
                    """
        _logger.info(epoch_msg)
        print(epoch_msg)

        loss_curve.update_values('train_loss', metrics['train_loss'])
        loss_curve.update_values('valid_loss', metrics['valid_loss'])
        loss_curve.plot_learning_curve(label='train_loss')
        loss_curve.plot_learning_curve(label='valid_loss')
        loss_curve.save_fig(f'./results/{args.exp_name}/loss_curve.png')
        
        acc_curve.update_values('train_acc', metrics['train_acc'])
        acc_curve.update_values('valid_acc', metrics['valid_acc'])
        acc_curve.plot_learning_curve(label='train_acc')
        acc_curve.plot_learning_curve(label='valid_acc')
        acc_curve.save_fig(f'./results/{args.exp_name}/acc_curve.png')

        if scheduler:
            scheduler.step()
    
        if args.save_ckpt:
            ckpt_dir = f'./results/{args.exp_name}/checkpoints'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch:04d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            _logger.info(f'Checkpoint saved at {ckpt_path}')


if __name__ == "__main__":
    parser = ArgumentParser()

    # experiment and log settings
    parser.add_argument('--neptune', action='store_true', help='Log results in Neptune')
    parser.add_argument('--save_ckpt', action='store_true', help="Save checkpoint at the end of every epoch.")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name")

    # datasets
    parser.add_argument('--datadir', '--data_dir', type=str, default='input/data/train/images', help='Data directory.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio. Set zero to use all data for training.')
    parser.add_argument('--shuffle_off', action='store_false', help="Do not shuffle train dataset.")

    # model configs
    parser.add_argument('--model', type=str, help='Name of the model to train.')
    parser.add_argument('--n_class', type=int, help="The number of classes.")
    parser.add_argument('--target', type=str, default='all', choices=['all', 'age', 'gender', 'mask'], help='Target label to predict.' )

    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'], help='Optimizer.')
    parser.add_argument('--scheduler', action='store_true', help='Use CosineAnnealingLR scheduler.')

    # system
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    main(args)
