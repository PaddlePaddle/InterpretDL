

import numpy as np
import os
import os.path as osp
import logging
import datetime
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import paddle
import paddle.nn.functional as F
from paddle.optimizer import Momentum
from paddle.optimizer.lr import MultiStepDecay

from paddle.vision.datasets import DatasetFolder
from paddle.vision import transforms
from paddle.vision.models import resnet101
from paddle.io import DataLoader


def load_train_test_datasets(dataset_root):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize = transforms.Resize(256)
    rcrop = transforms.RandomCrop((224, 224))
    ccrop = transforms.CenterCrop((224, 224))
    tot = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)

    train_transforms = transforms.Compose([resize, rcrop, tot, normalize])
    test_transforms = transforms.Compose([resize, ccrop, tot, normalize])

    train_set = DatasetFolder(osp.join(dataset_root, 'train'), transform=train_transforms)
    test_set = DatasetFolder(osp.join(dataset_root, 'test'), transform=test_transforms)

    return train_set, test_set


def evaluate(model, loader):
    fin_targets = []
    fin_probabs = []
    
    with paddle.no_grad():
        model.eval()
        epoch_loss = 0
        for batch in tqdm(loader, unit="batches", desc="Evaluating"):
            
            imgs, targets = batch

            logits = model(imgs)

            loss = F.cross_entropy(logits, targets)
            epoch_loss += loss.item() * loader.batch_size

            fin_targets.extend(targets.tolist())
            fin_probabs.extend(F.softmax(logits, axis=1).numpy())

        loss = epoch_loss / len(loader.dataset)
        fin_preds = np.argmax(np.array(fin_probabs), axis=-1)

    acc = accuracy_score(fin_targets, fin_preds)
    return loss, acc


def run(args, train_set, test_set):
    model = resnet101(pretrained=True, num_classes=len(train_set.classes))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

    no_decay_params = []
    decay_params = []
    for n, v in model.named_parameters():
        if 'norm' in n or 'bias' in n:
            no_decay_params.append(v)
        else:
            decay_params.append(v)

    list_params = [{'params': decay_params, 'weight_decay': args.wd}, {'params': no_decay_params, 'weight_decay': 0.0}]
    
    scheduler = MultiStepDecay(learning_rate=args.lr, milestones=[int(args.epochs*0.67)], gamma=0.1)
    optimizer = Momentum(scheduler, parameters=list_params)

    logging.info("Training Started...")
    logging.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    for epoch in range(1, args.epochs+1, 1):
        model.train()
        bar = tqdm(train_loader)
        bar.set_description("Training")
        losses = 0
        for idx, batch in enumerate(bar):
            imgs, targets = batch

            logits = model(imgs)

            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses += loss.item() * train_loader.batch_size
            acc = (logits.argmax(1) == targets).cast('float32').mean()

            if idx % 100 == 0:
                logging.info(
                    f" EPOCH| {epoch}, BATCH| {idx}/{len(bar)}, LOSS| {loss.item(): .4f}, ACC| {acc.item(): .4f}"
                )
        
        logging.info(f" EPOCH| {epoch}, BATCH| {idx}/{len(bar)}, LOSS| {loss.item(): .4f}, ACC| {acc.item(): .4f}")
        scheduler.step()

        logging.info(f"EPOCH | {epoch} TOTAL LOSS | {losses / len(train_loader.dataset): .4f}")

        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f"EPOCH | {epoch} TOTAL LOSS | {losses / len(train_loader.dataset): .4f}")

        # writer.add_scalar('Loss/train', losses / len(train_loader.dataset), epoch)
        
        loss_test, acc_test = evaluate(model, test_loader)
        logging.info(f"TEST ACC| {acc_test}, LOSS| {loss_test}")
        print(f"TEST ACC| {acc_test}, LOSS| {loss_test}")
        
        if epoch % args.ckpt == 0:
            paddle.save(model.state_dict(), f'./work_dirs/result_{args.name}/ckpt-{epoch}.pd')

    loss_test, acc_test = evaluate(model, test_loader)
    logging.info(f"TEST ACC| {acc_test}, LOSS| {loss_test}")
    print(f"TEST ACC| {acc_test}, LOSS| {loss_test}")
    
    paddle.save(model.state_dict(), f'./work_dirs/result_{args.name}/ckpt-final.pd')


def main(args):

    train_set, val_day_set = load_train_test_datasets(args.dataset_dir)
    run(args, train_set, val_day_set)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='fine-tuning', type=str, help="exp name")
    parser.add_argument('--dataset_dir', default="/root/datasets/CUB_200_2011/", type=str, help="dataset_dir")
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('--wd', default=1e-4, type=float, help="weight decay")
    parser.add_argument('--random_seed', default=77, type=int, help="lucky number")
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to train.')
    parser.add_argument('--ckpt', default=30, type=int, help='Number of epochs to save a ckpt.')
    parser.add_argument('--log', default="INFO", help="Logging level.")
    parser.add_argument('--device', default='gpu:0', type=str, help="gpu:0, cpu")

    args = parser.parse_args()
    
    device = args.device
    paddle.set_device(device)
    np.random.seed(args.random_seed)
    
    if not os.path.exists('./work_dirs'):
        os.makedirs('./work_dirs')
    if not os.path.exists(f'./work_dirs/result_{args.name}'):
        os.makedirs(f'./work_dirs/result_{args.name}')
    FORMAT = '%(asctime)-15s %(message)s'
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        filename=f'./work_dirs/result_{args.name}/app-{time_stamp}.log', 
        filemode='w', 
        format=FORMAT,
        level=getattr(logging, args.log.upper())
    )
    logging.info(f'{args}\n')

    main(args)

