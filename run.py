import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, subset_cifar
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from torch.utils.data import RandomSampler
from tqdm import tqdm
from models.resnet_simclr import Linear_classify
from utils import load_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


# cached_model_name

parser.add_argument('--cached_model_name', default=None, type=str,
                    help='initial learning rate')

parser.add_argument('--test', action='store_true',
                    help='Disable CUDA')

# args.output_dir

parser.add_argument('--output_dir', default=None, type=str,
                    help='initial learning rate')

parser.add_argument('--meta_lr', default=20, type=float,
                    help='initial learning rate')

parser.add_argument('--meta', action='store_true',
                    help='initial learning rate')

parser.add_argument('--joint_training', action='store_true',
                    help='initial learning rate')


# joint_training

def validate(test_loader, model, linear_classifier, criterion, epoch, args):
    model.eval()
    linear_classifier.eval()

    with torch.no_grad():
        total_loss = 0
        total_count = 0
        correct_count = 0
        for _, images, labels in tqdm(test_loader):
            if type(images) is tuple:
                images = images[0]
            images = images.to(args.device)
            labels = labels.to(args.device)
            model_feature = model(images)
            model_out = linear_classifier(model_feature)
            total_count += images.shape[0]
            pred_labels = model_out.max(1)[1]
            correct_count += torch.sum(labels.view(-1) == pred_labels.view(-1)).detach().cpu().item()
            loss = criterion(model_out, labels)
            total_loss += loss.item()
        test_acc = correct_count/total_count
        test_loss = total_loss/total_count

        
        print("test accuracy::", test_acc)
        print("test loss::", test_loss)

def train_classifer(train_loader, valid_loader, test_loader, model, criterion, args):

    linear_classifier = Linear_classify(args.feature_dim, args.num_classes)

    linear_classifier = linear_classifier.to(args.device)

    if args.joint_training:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(linear_classifier.parameters()), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(linear_classifier.parameters(), args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train_classifer_one_epoch(train_loader, model, linear_classifier, criterion, optimizer, epoch, args)    

        
        print("Validation performance at epoch ", epoch)
        validate(valid_loader, model, linear_classifier, criterion, epoch, args)

        print("Test performance at epoch ", epoch)
        validate(test_loader, model, linear_classifier, criterion, epoch, args)

def train_classifer_one_epoch(train_loader, model, linear_classifier, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.train()
    linear_classifier.train()
    # classifier.train()

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # losses1 = AverageMeter()
    # losses2 = AverageMeter()
    # top1 = AverageMeter()

    # end = time.time()
    # for idx, items in enumerate(train_loader):
    correct_items = 0
    total_items = 0
    total_loss = 0
    for train_ids, images, labels in tqdm(train_loader):
        # image_ls = []
        # for idx in range(args.n_views):
        #     image_ls.append(images[idx].to(args.device))
        # images = torch.cat(image_ls, dim=0)
        # images = images[0]
        images = images.to(args.device)
        labels = labels.to(args.device)

        if not args.joint_training:
            with torch.no_grad():
                model_feature = model(images)
        else:
            model_feature = model(images)

        model_out = linear_classifier(model_feature)

        pred_labels = model_out.max(1)[1]

        loss = criterion(model_out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_items += torch.sum(labels.view(-1) == pred_labels.view(-1)).detach().cpu().item()
        total_items += labels.view(-1).shape[0]
        # optimizer.second_step(zero_grad=True)

    average_loss = total_loss/total_items
    train_acc = correct_items/total_items
    print("training performance at epoch ", epoch)
    print("average train loss::", average_loss)
    print("average train accuracy::", train_acc)






def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    if not args.test:
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

        test_dataset = None
    else:
        train_dataset = dataset.get_eval_dataset(args.dataset_name, True)

        test_dataset = dataset.get_eval_dataset(args.dataset_name, False)

    train_dataset, valid_dataset = subset_cifar(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    meta_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=args.epochs*len(train_loader)*args.batch_size*10)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=(meta_sampler is None),pin_memory=True, sampler = meta_sampler)

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.test:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)


    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    if args.test:
        load_checkpoint(model, filename=args.cached_model_name)
        args.num_classes = 10
        args.feature_dim = list(model.backbone.fc)[0].in_features
        model.backbone.fc = torch.nn.Identity()
        # model.fc = torch.nn.Linear(model.backbone.fc.in_features, args.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    if not args.test:
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            if not args.meta:
                simclr.train(train_loader)
            else:
                simclr.meta_train(train_loader, valid_loader)
    else:
        
        
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        model = model.to(args.device)
        train_classifer(train_loader, valid_loader, test_loader, model, criterion, args)



if __name__ == "__main__":
    main()
