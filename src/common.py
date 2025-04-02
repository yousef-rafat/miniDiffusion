import torch
import torch.nn as nn

##################################
# https://arxiv.org/pdf/2403.03206
##################################

def loss_fn(v_pred: torch.Tensor, v_true: torch.Tensor, alpha_t):
    # computes MSE for velocity vectors with scaled alpha t over beta t

    alpha_t = torch.sqrt(alpha_t)
    beta_t = torch.sqrt(1 - alpha_t)

    # small value to avoid division by zero
    weight = (alpha_t / (beta_t + 1e-8))

    v_pred = v_pred[:, :, :v_true.size(-1), :v_true.size(-1)]
    
    loss = (weight * (v_pred - v_true) ** 2).mean()

    return loss

def interpolate_samples(source, target, t):
    # linearly interpolate latents between noise and data
    return source * (1 - t) + target * t

def get_ground_truth_velocity(image: torch.Tensor, noise: torch.Tensor, alpha_t):
    # refer to reference papar above:

    a_t = torch.sqrt(alpha_t)
    b_t = torch.sqrt(1 - alpha_t)

    v_t = (a_t * image + b_t * noise) / torch.sqrt(a_t**2 + b_t**2)
    return v_t

def compute_clip_loss(clip, generated_image, input_ids, attention_mask, size: int):
    " compute the contrasive loss between text and generated image "

    # adjut image size (important in case of rounding dimensions in VAE)
    clip.adjust_image_size(size)

    # get logits
    logits = clip(generated_image, input_ids, attention_mask)

    if logits.size(0) < 2: raise ValueError("clip loss requires at least two samples in batch")
    
    labels = torch.arange(logits.size(0), device = logits.device).long()

    # compute the loss in two ways
    loss_i2t = nn.CrossEntropyLoss()(logits, labels) # image to text
    loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels) # text to image

    # average the two losses
    loss = (loss_i2t + loss_t2i) / 2

    return loss

def test_clip_loss():
    label = "cats have friends"#torch.rand(2, 512)
    image = torch.rand(2, 3, 222, 222)
    
    from model.clip import CLIP
    from model.dit_components import HandlePrompt

    label, attention_mask = HandlePrompt()(label)
    label = label.long().permute(1, 0, 2).expand(2, -1, -1)
    attention_mask = attention_mask.unsqueeze(0).expand(2, -1)
    
    loss = compute_clip_loss(clip = CLIP(),generated_image = image, input_ids = label, attention_mask = attention_mask, size = image.size(-1)) #torch.zeros_like(label)

    print(loss)