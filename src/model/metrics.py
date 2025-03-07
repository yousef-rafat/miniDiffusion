import torch
import torch.nn as nn
import torch.functional as F

class FID(nn.Module):
    # reference equation: https://www.oreilly.com/library/view/generative-adversarial-networks/9781789136678/9bf2e543-8251-409e-a811-77e55d0dc021.xhtml
    def __init__(self):
        super(FID, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = True)

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

        with torch.no_grad():
            gen_features = self.model(gn_img)
            rl_features = self.model(rl_img)

        # get the mean and the covariance matrix out of the output
        mu_gen = torch.mean(gen_features, dim = 0)
        mu_rl = torch.mean(rl_features, dim = 0)

        sigma_gen = torch.cov(gen_features)
        sigma_rl = torch.cov(rl_features)

        # compute square difference
        diff = mu_rl - mu_gen
        mu_diff = torch.sum(diff ** 2)

        covmean = self.matrix_sqrt(sigma_gen = sigma_gen, sigma_rl = sigma_rl)

        fid_score = mu_diff + torch.trace(sigma_gen + sigma_rl - 2 * covmean)

        return fid_score