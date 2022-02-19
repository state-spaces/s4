#!/usr/bin/env python
"""Train a CNN for Google speech commands."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'
"""With modifications from Karan Goel."""

import argparse
import time

import torch
import torchvision

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from models.resnext import CifarResNeXt
from speech_commands_dataset import BackgroundNoiseDataset, CLASSES, SpeechCommandsDataset
from transforms import (
    AddBackgroundNoiseOnSTFT, 
    ChangeAmplitude, 
    ChangeSpeedAndPitchAudio, 
    DeleteSTFT,
    FixAudioLength, 
    FixSTFTDimension,
    LoadAudio,
    StretchAudioOnSTFT, 
    TimeshiftAudioOnSTFT, 
    ToMelSpectrogramFromSTFT,
    ToMelSpectrogram,
    ToSTFT, 
    ToTensor,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

n_mels = 32
if args.input == 'mel40':
    n_mels = 40

data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform)
add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
train_dataset = SpeechCommandsDataset(args.train_dataset,
                                Compose([LoadAudio(),
                                         data_aug_transform,
                                         add_bg_noise,
                                         train_feature_transform]))

valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
valid_dataset = SpeechCommandsDataset(args.valid_dataset,
                                Compose([LoadAudio(),
                                         FixAudioLength(),
                                         valid_feature_transform]))

weights = train_dataset.make_weights_for_balanced_classes()
sampler = WeightedRandomSampler(weights, len(weights))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


# a name used to save checkpoints etc.
full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % ('resnext', args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
if args.comment:
    full_name = '%s_%s' % (full_name, args.comment)

model = CifarResNeXt(nlabels=len(CLASSES), in_channels=1)

if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

criterion = torch.nn.CrossEntropyLoss()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

start_timestamp = int(time.time()*1000)
start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0

if args.resume:
    print("resuming a checkpoint '%s'" % args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_accuracy = checkpoint.get('accuracy', best_accuracy)
    best_loss = checkpoint.get('loss', best_loss)
    start_epoch = checkpoint.get('epoch', start_epoch)
    global_step = checkpoint.get('step', global_step)

    del checkpoint  # reduce memory

if args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

def get_lr():
    return optimizer.param_groups[0]['lr']

writer = SummaryWriter(comment=('_speech_commands_' + full_name))

def train(epoch):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase,  get_lr(), epoch)

    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward/backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item() # loss.data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

def valid(epoch):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer' : optimizer.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s.pth' % full_name)
        torch.save(model, '%d-%s-best-loss.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, 'checkpoints/best-acc-speech-commands-checkpoint-%s.pth' % full_name)
        torch.save(model, '%d-%s-best-acc.pth' % (start_timestamp, full_name))

    torch.save(checkpoint, 'checkpoints/last-speech-commands-checkpoint.pth')
    del checkpoint  # reduce memory

    return epoch_loss

print("training %s for Google speech commands..." % 'resnext')
since = time.time()
for epoch in range(start_epoch, args.max_epochs):
    if args.lr_scheduler == 'step':
        lr_scheduler.step()

    train(epoch)
    epoch_loss = valid(epoch)

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=epoch_loss)

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
    print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
print("finished")
