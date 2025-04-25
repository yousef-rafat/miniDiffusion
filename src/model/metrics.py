import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights

class FID(nn.Module):
    # reference equation: https://www.oreilly.com/library/view/generative-adversarial-networks/9781789136678/9bf2e543-8251-409e-a811-77e55d0dc021.xhtml
    def __init__(self):
        super(FID, self).__init__()

        os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "encoders")

        model_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "inception_v3_google-0cc3c7bd.pth")

        if not os.path.exists(model_path): self.model = torchvision.models.inception_v3(weights = Inception_V3_Weights.IMAGENET1K_V1)
        else: 
            # initate model and load state
            self.model = torchvision.models.inception_v3(init_weights = False)
            self.model.load_state_dict(torch.load(model_path))

        # perform no operation and return the features
        self.model.fc = nn.Identity()

        self.model.eval()

    def matrix_sqrt(self, sigma_gen: torch.Tensor, sigma_rl: torch.Tensor) -> torch.Tensor:
        X = sigma_gen @ sigma_rl # mat product

        # get eigenvector and values
        eigenvalues, eigenvectors = torch.linalg.eigh(X) 

        # eigenvalues may have negative values due to numerical issues
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min = 0))

        covmean = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.t()

        return covmean

    def forward(self, generated_image: torch.Tensor, real_image: torch.Tensor):

        # resize images into 299 for inception
        gn_img = F.interpolate(generated_image, size = (299, 299), mode = 'bilinear' , align_corners = False)
        rl_img = F.interpolate(real_image, size = (299, 299), mode = 'bilinear' , align_corners = False)

        # normalize images
        gn_img = gn_img / 255.0

        with torch.no_grad():
            gen_features = self.model(gn_img)
            rl_features = self.model(rl_img)

        # get the mean and the covariance matrix out of the output
        mu_gen = torch.mean(gen_features, dim = 0)
        mu_rl = torch.mean(rl_features, dim = 0)

        # covariance matrices
        # avoid outputing scalars (0D)
        sigma_gen = torch.cov(gen_features.T) if gen_features.shape[0] > 1 else torch.eye(gen_features.shape[1])
        sigma_rl = torch.cov(rl_features.T) if rl_features.shape[0] > 1 else torch.eye(rl_features.shape[1])

        eps = 1e-6
        # add a small eps to the diagonal of the sigmas (better num. stability)
        sigma_gen += torch.eye(sigma_gen.size(0)).to(sigma_gen.device) * eps
        sigma_rl += torch.eye(sigma_rl.size(0)).to(sigma_rl.device) * eps

        # compute square difference
        diff = mu_rl - mu_gen
        mu_diff = torch.sum(diff ** 2)

        covmean = self.matrix_sqrt(sigma_gen = sigma_gen, sigma_rl = sigma_rl)

        fid_score = mu_diff + torch.trace(sigma_gen + sigma_rl - 2 * covmean)

        return torch.clamp(fid_score, min = 0)
    
def test_fid():
    gen_img = torch.rand(3, 224, 224)
    img = torch.rand(3, 224, 224)

    fid = FID()
    score = fid(gen_img.unsqueeze(0), img.unsqueeze(0))
    print(score)