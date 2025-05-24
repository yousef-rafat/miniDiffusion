import math
import torch
from torch.nn.functional import cosine_similarity

##################################
# https://arxiv.org/pdf/2403.03206
##################################

def logit_normal_weighting(t: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:

    """ Logit-Normal Weighting
        w(t) = 1/pi_ln(t; mean, std)
        from Eq. (19) in the SD3 paper.
    """

    # avoid 0/1 edges
    eps = 1e-6
    t = t.clamp(eps, 1 - eps)

    # weight w(t) equals the reciprocal of the pdf function (to remove bias)
    term1 = (t * (1 - t)) * (std * math.sqrt(2 * math.pi))

    logit_t = (t / (1 - t)).log()
    term2 = torch.exp((logit_t - mean) ** 2 / (2 * (std ** 2)))

    output = term1 * term2
    
    return output

def loss_fn(v_pred: torch.Tensor, v_true: torch.Tensor, sigma):
    # computes MSE for velocity vectors with logit normal weighting

    weight = logit_normal_weighting(sigma)
    
    loss = (weight * (v_pred - v_true) ** 2)\
        .reshape(v_pred.size(0), -1).mean(dim = 1)

    return loss.mean()

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