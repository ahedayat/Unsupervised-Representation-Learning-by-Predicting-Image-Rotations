"""
    'training' and 'evaluation' module is implemented in this file.
"""

import os
from tqdm import tqdm
import pandas as pd

import torch
from torch.autograd import Variable

import nets as nets
from utils import AverageMeter
from .dl_utils import accuracy


def net_train(
    model,
    train_data_loader,
    val_data_loader,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    batch_size=64,
    num_workers=2,
    Tensor=torch.Tensor,
    report_path="./reports",
    saving_checkpoint_path="./checkpoints",
    saving_prefix="ckpt_",
    saving_checkpoint_freq=20
):
    """
        In this function, train a classifier on features learned in 
        self-supervised manner. 
    """

    train_data_loader = torch.utils.data.DataLoader(
        train_data_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_data_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )

    for epoch in range(num_epochs):

        top1 = AverageMeter()
        top2 = AverageMeter()
        losses = AverageMeter()

        train_report = pd.DataFrame({
            "mode": [],
            "epoch": [],
            "batch_id": [],
            "batch_size": [],
            "loss": [],
            "acc1": [],
            "acc2": [],
            "lr": []
        })

        model.train()

        with tqdm(train_data_loader) as t_train_data_loader:
            for it, (X, Y) in enumerate(t_train_data_loader):
                t_train_data_loader.set_description(
                    f"Training Classifier @ Epoch {epoch}")

                X = Variable(X.type(Tensor))
                Y = Variable(Y.type(torch.LongTensor))
                if X.device.type == "cuda":
                    Y = Y.cuda()

                out = model(X)

                loss = criterion(out, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc1, acc2 = accuracy(out, Y, topk=(1, 2))
                losses.update(loss.item(), X.size(0))
                top1.update(acc1, X.size(0))
                top2.update(acc2, X.size(0))

                t_train_data_loader.set_postfix(
                    loss="{:.3f}".format(losses.avg),
                    acc1="{:.3f}".format(top1.avg),
                    acc2="{:.3f}".format(top2.avg),
                    lr="{:.5e}".format(optimizer.param_groups[0]["lr"])
                )

                train_report = train_report.append(
                    {
                        "mode": "train",
                        "epoch": epoch,
                        "batch_id": it,
                        "batch_size": X.shape[0],
                        "loss": loss.item(),
                        "acc1": acc1,
                        "acc2": acc2,
                        "lr": optimizer.param_groups[0]["lr"]
                    }, ignore_index=True
                )

        # Saving Training Report
        train_report.to_csv(os.path.join(
            report_path, f"train_{epoch}.csv"))

        # Evaluating classifier with validation dataset
        val_loss, val_acc = net_eval(
            model=model,
            eval_data_loader=val_data_loader,
            criterion=criterion,
            epoch=epoch,
            Tensor=Tensor,
            report_path=report_path,
            report_name=f"val_{epoch}"
        )

        lr_scheduler.step(val_loss)

        if ((epoch+1) % saving_checkpoint_freq) == 0:
            nets.save(
                file_path=saving_checkpoint_path,
                file_name=f"{saving_prefix}{epoch}.ckpt",
                model=model.backbone,
                optimizer=optimizer,
            )
            print(f"(Epoch: {epoch}) > Model Saved.")

    return model, optimizer


def net_eval(
    model,
    eval_data_loader,
    criterion,
    Tensor=torch.Tensor,
    report_path="./reports",
    report_name="test",
    epoch=None
):
    """
        In this function, evaluating procedure is implemeted.
    """
    val_report = pd.DataFrame({
        "mode": [],
        "epoch": [],
        "batch_id": [],
        "batch_size": [],
        "loss": [],
        "acc1": [],
        "acc2": []
    })

    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()

    with tqdm(eval_data_loader) as t_eval_data_loader:
        for it, (X, Y) in enumerate(t_eval_data_loader):
            if epoch is not None:
                t_eval_data_loader.set_description(
                    f"Evaluation Classifier @ Epoch {epoch}")
            else:
                t_eval_data_loader.set_description(
                    f"Evaluation Classifier")
                    
            X = Variable(X.type(Tensor))
            Y = Variable(Y.type(torch.LongTensor))
            if X.device.type == "cuda":
                Y = Y.cuda()

            out = model(X)

            loss = criterion(out, Y)

            acc1, acc2 = accuracy(out, Y, topk=(1, 2))
            losses.update(loss.item(), X.size(0))
            top1.update(acc1, X.size(0))
            top2.update(acc2, X.size(0))

            val_report = val_report.append(
                {
                    "mode": "val",
                    "epoch": epoch,
                    "batch_id": it,
                    "batch_size": X.shape[0],
                    "loss": loss.item(),
                    "acc1": acc1,
                    "acc2": acc2
                }, ignore_index=True
            )

            t_eval_data_loader.set_postfix(
                loss="{:.3f}".format(losses.avg),
                acc1="{:.3f}".format(top1.avg),
                acc2="{:.3f}".format(top2.avg),
            )

    val_report.to_csv(os.path.join(report_path, f"{report_name}.csv"))

    return losses.avg, top1.avg
