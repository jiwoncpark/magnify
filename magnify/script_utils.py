import os
import torch


def save_state(model, optim, lr_scheduler, kl_scheduler, epoch,
               train_dir, param_w_scheduler, epoch_i):
    """Save the state dict of the current training to disk
    Parameters
    ----------
    train_loss : float
        current training loss
    val_loss : float
        current validation loss
    """
    state = dict(
             model=model.state_dict(),
             optimizer=optim.state_dict(),
             lr_scheduler=lr_scheduler.state_dict(),
             kl_scheduler=kl_scheduler.__dict__,
             param_w_scheduler=param_w_scheduler.__dict__,
             epoch=epoch,
             )
    model_path = os.path.join(train_dir, f'model_{epoch_i}.mdl')
    torch.save(state, model_path)


def load_state(model, train_dir, device,
               optim=None, lr_scheduler=None, kl_scheduler=None,
               param_w_scheduler=None,
               epoch_i=0,
               ):
    """Load the state dict to resume training or infer

    """
    model_path = os.path.join(train_dir, f'model_{epoch_i}.mdl')
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    model.to(device)
    if optim is not None:
        optim.load_state_dict(state['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state['lr_scheduler'])
    if kl_scheduler is not None:
        kl_scheduler.__dict__ = state['kl_scheduler']
    if param_w_scheduler is not None:
        param_w_scheduler.__dict__ = state['param_w_scheduler']
    print(f"Loaded model at epoch {state['epoch']}")
