import torch

def get_ground_truth_velocity(x0: torch.Tensor, xT: torch.Tensor):
    " Function computes the ground truth velocity vector "

    # x0 -> ground truth (original image)
    # xT -> noise  

    v_true = x0 - xT
    return v_true

def loss_fn(v_pred, v_true):
    # computes MSE for velocity vectors
    loss = ((v_pred - v_true) ** 2).mean()
    return loss

