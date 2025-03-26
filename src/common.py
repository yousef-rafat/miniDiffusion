import torch
import torch.nn as nn

##################################
# https://arxiv.org/pdf/2403.03206
##################################

def loss_fn(v_pred: torch.Tensor, v_true: torch.Tensor, alpha_t):
    # computes MSE for velocity vectors with scaled alpha t over beta t

    alpha_t = torch.sqrt(alpha_t)
    beta_t = torch.sqrt(alpha_t - 1)

    # small value to avoid division by zero
    weight = (alpha_t / (beta_t + 1e-8))

    loss = (weight * (v_pred - v_true) ** 2).mean()

    return loss

def interpolate_samples(source, target, t):
    # linearly interpolate samples
    return source * (1 - t) + target * t

def get_ground_truth_velocity(image: torch.Tensor, noise: torch.Tensor, t,  alpha_t):
    # refer to reference papar above:

    alpha_t = alpha_t[t]

    a_t = torch.sqrt(alpha_t)
    b_t = torch.sqrt(1 - alpha_t)

    v_t = (a_t * image + b_t * noise) / torch.sqrt(a_t**2 + b_t**2)
    return v_t

def compute_clip_loss(clip, generated_image, input_ids, attention_mask):
    " compute the contrasive loss between text and generated image "

    # get logits
    logits = clip(generated_image, input_ids, attention_mask)

    if logits.size(0) < 2: raise ValueError("clip loss requires at least two samples in batch")
    
    labels = torch.arange(logits.size(0), device = logits.device)

    # compute the loss in two ways
    loss_i2t = nn.CrossEntropy()(logits, labels) # image to text
    loss_t2i = nn.CrossEntropy()(logits.t(), labels) # text to image

    # average the two losses
    loss = (loss_i2t + loss_t2i) / 2

    return loss