import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

##################################
# https://arxiv.org/pdf/2403.03206
##################################

def loss_fn(v_pred: torch.Tensor, v_true: torch.Tensor, alpha_t):
    # computes MSE for velocity vectors with scaled alpha t over beta t

    sqrt_alpha_t = torch.sqrt(alpha_t)
    beta_t = torch.sqrt(1 - alpha_t)

    # small value to avoid division by zero
    weight = (sqrt_alpha_t / (beta_t + 1e-8))

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

def compute_clip_loss(clip, generated_image, text_embs):
    " compute the similarity loss between text and generated image "

    # get features
    with torch.no_grad():
        image_features = clip.encode_image(generated_image)
        text_features, _ = clip.encode_text(text_embs)

    sim = cosine_similarity(image_features, text_features, dim = -1).mean()
    loss = 1 - sim

    return loss

def test_clip_loss():

    label = ["cat", "dog", "kettle"] #torch.rand(1, 512)

    from PIL import Image
    from torchvision.transforms import ToTensor
    from torchvision.transforms.v2 import Resize
    import os
 
    image_path = os.path.join(os.getcwd(), "assets", "cat.webp")

    image = Image.open(image_path).convert("RGB")

    image = Resize(size=(224, 224))(image)

    image = ToTensor()(image).unsqueeze(0)
    
    from model.clip import CLIP, load_clip

    model = CLIP()
    model = load_clip(model)
    for l in label:
        loss = compute_clip_loss(clip = model, generated_image = image, text_embs = l)
        print(f"Loss for {l}:", loss)

if __name__ == "__main__":
    test_clip_loss()