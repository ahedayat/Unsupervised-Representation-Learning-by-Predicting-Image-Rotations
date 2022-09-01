import os
import math
import utils as utils
from datetime import datetime, date


import torch
from torchvision import transforms, models

import optim
import nets as nets
import deeplearning as dl
import dataloaders as data


def save_report(df, backbone_name, saving_path):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "{}_{}_{}_{}_{}_{}_{}.csv".format(
        backbone_name, year, month, day, hour, minute, second)

    print("Saving Report at '{}'".format(
        os.path.join(saving_path, report_name)))

    df.to_csv(os.path.join(saving_path, report_name))


def get_data_loaders(args, mean, std, rotation_angles=(0, 90, 180, 270), train_ratio=0.8):
    """
        Returning train, validation and test set
    """

    # - Train
    train_data_loader = data.Cifar10Rotation(
        data_path=args.train_data_path,
        data_mode="train",
        input_normalization=True,
        mean=mean,
        std=std,
        rotation_angles=rotation_angles,
        data_download=True,
        show_all_rotations=args.show_all_angles,
        shuffle=args.shuffle_data
    )

    # - Validation
    val_data_loader = data.Cifar10Rotation(
        data_path=args.val_data_path,
        data_mode="val",
        input_normalization=True,
        mean=mean,
        std=std,
        rotation_angles=rotation_angles,
        data_download=True,
        show_all_rotations=False,
        shuffle=False
    )

    # - Test
    test_data_loader = data.Cifar10Rotation(
        data_path=args.test_data_path,
        data_mode="test",
        input_normalization=True,
        mean=mean,
        std=std,
        rotation_angles=rotation_angles,
        data_download=True,
        show_all_rotations=False,
        shuffle=False
    )

    # Train and Validation Spliting
    train_samples_index = train_data_loader.samples_index

    train_lower_index = 0
    train_upper_index = math.floor(train_ratio * len(train_samples_index))
    val_upper_index = len(train_samples_index)

    train_data_loader.samples_index = train_samples_index[train_lower_index: train_upper_index]
    val_data_loader.samples_index = train_samples_index[train_upper_index: val_upper_index]

    return train_data_loader, val_data_loader, test_data_loader


def _main(args):

    print("#"*32)
    for arg in str(args).split(","):
        print(arg)

    print("#"*32)

    # Hardware
    cuda = True if args.gpu and torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Dataloders
    mean, std = (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
    rotation_angles = [0, 90, 180, 270]

    data_loaders = get_data_loaders(
        args,
        mean=mean,
        std=std,
        rotation_angles=rotation_angles,
        train_ratio=0.9
    )

    train_data_loader, val_data_loader, _ = data_loaders

    print("Number of training data: ", len(train_data_loader))
    print("Number of validation data: ", len(val_data_loader))

    # CNN Backbone
    assert args.backbone in [
        "resnet34", "NIN"], "Backbone network must be one of this items: ['resnet34', 'NIN']"

    num_classes = 4
    model = None
    if args.backbone == "NIN":
        model = nets.NetworkInNetwork(num_classes=num_classes)
    elif args.backbone == "resnet34":
        model = nets.MyResNet34(num_classes=num_classes,
                                pretrained=True, freeze_backbone=False)
        model.unfreeze_backbone()

    # Optimizer
    assert args.optimizer in [
        "sgd", "adam"], "Optimizer must be one of this items: ['sgd', 'adam']"

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(optimizer)

    # Learning Rate Schedular
    lr_scheduler = optim.lr_scheduler(
        optimizer,
        decay_epochs=(30, 60, 80),
        # decay_epochs=(2, 4),
        decay_coefficient=0.2
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='max',
                                                              patience=3,
                                                              threshold=0.9,
                                                              min_lr=1e-6,
                                                              verbose=False,
                                                              )

    # Loading Model
    if args.ckpt_load_path is not None:
        print("**** Loading Model...")
        net, optimizer = nets.load(
            ckpt_path=args.ckpt_load_path, model=model, optimizer=optimizer)

    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Checkpoint Address
    saving_path, saving_prefix = args.ckpt_save_path, args.ckpt_prefix
    saving_checkpoint_freq = args.ckpt_save_freq

    # Training
    model, optimizer = dl.train(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        report_path=args.report_path,
        Tensor=Tensor,
        saving_checkpoint_path=saving_path,
        saving_prefix=saving_prefix,
        saving_checkpoint_freq=saving_checkpoint_freq
    )

    nets.save(
        file_path=saving_path,
        file_name=f"{saving_prefix}final.ckpt",
        model=model.backbone,
        optimizer=optimizer
    )


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
