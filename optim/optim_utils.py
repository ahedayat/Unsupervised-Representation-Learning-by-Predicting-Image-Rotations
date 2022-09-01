import torch


def lr_scheduler(optimizer, decay_epochs=(30, 60, 80), decay_coefficient=0.2):
    """
        This function returns learning rate scheduler
    """
    if type(decay_epochs) != list:
      decay_epochs = list(decay_epochs)

    def lmabda_lr_func(epoch):
        decay_epochs.sort()

        pow = 0
        for e in decay_epochs:
            if epoch >= e:
                pow += 1
            else:
                break
        return decay_coefficient ** pow

    _lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lmabda_lr_func)

    return _lr_scheduler
