import torch

# Euler Scheduler For Training and Inference
#////////////////////////////////////////////////////////////
class NoiseScheduler(torch.nn.Module):
    def __init__(self, num_training_timesteps: int = 1_000, base_shift: float = 0.5, shift: float = 3, 
                 num_inference_timesteps: int = 50, inference: bool = False):
        super(NoiseScheduler, self).__init__()

        # shift normal dist left or right (mean)
        self.base_shift = base_shift
        # control shape of logit-normal distrubition (standard deviation)
        # shift < 1 = decay the noise earlier
        self.shift = shift

        # compute timestep values so we can index into them later
        timesteps = torch.linspace(1.0, num_training_timesteps, int(num_training_timesteps))

        # normalize between 0 and 1
        sigmas = timesteps / num_training_timesteps

        # staticaly shift (fixed image size assumed)
        self.sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)

        # get timesteps after shifting
        self.timesteps = self.sigmas * num_training_timesteps

        self.num_training_timesteps = num_training_timesteps
        self.num_inference_timesteps = num_inference_timesteps

        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        if inference:

            max_timestep = self.sigma_to_timestep(self.sigma_max)
            min_timestep = self.sigma_to_timestep(self.sigma_min)

            # go from full noise to clean image (descending)
            timesteps = torch.linspace(max_timestep, min_timestep, num_inference_timesteps)

            # same as for training
            sigmas = timesteps / num_training_timesteps
            sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
            timesteps = sigmas * num_training_timesteps

            # add a zero at the end to reach the data distribution
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device = sigmas.device)])

        self.step_index = 0

    def sigma_to_timestep(self, sigma):
        return sigma * self.num_training_timesteps

    def check_timestep(self, timestep):
        # check if timestep is valid

        if isinstance(timestep, torch.Tensor):
            timestep = float(timestep.item())
            
        if timestep > self.num_training_timesteps:
            raise ValueError("Can't have a timestep larger than the defined timestep")

    
    def get_sigmas(self, timesteps, n_dim: int = 4):
        " From the timestep, get the corresponding sigma (noise scale) "

        # use binary search (torch.searchsorted) for finding indices
        step_indices = [torch.searchsorted(self.timesteps, t).item() for t in timesteps]

        sigma = self.sigmas[step_indices].flatten()
        
        # expand to n_dim dims
        sigma = sigma.view(tuple([sigma.size(0)] + [1] * (n_dim - 1)))

        return sigma
    
    def index_of_timestep(self, timestep):
        indice = (self.timesteps == timestep).nonzero()
        return indice[0].item()
        
    def sample_logit_timestep(self, batch_size: int = 1, device: str = "cpu") -> float:
        " Sample timesteps from logit normal distribution "
        # returns timesteps in range [0, timesteps - 1]
        # beneficial to train the model on different noise levels

        uniform_samples = torch.rand(batch_size, device = device)
        # turn a uniform sample into a normal sample
        normal_samples = torch.logit(uniform_samples, eps = 1e-8) * self.shift + self.base_shift

        sampled_t = torch.sigmoid(normal_samples)

        weighted_sampled_t = (sampled_t * (self.num_training_timesteps - 1)).long() # scale
        
        return weighted_sampled_t.item()
        

    def add_noise(self, image: torch.FloatTensor,  timestep: float = None):

        """
        Forward Process of Diffusion
        Adds noise to the image according to the timestep.
        t = Timestep at which to evaluate the noise schedule.
        """

        if timestep is None:
            timestep = self.sample_logit_timestep(image.size(0), device = image.device)

        self.check_timestep(timestep) 

        # randn_like will create the guassian noise
        # that will fit the image's dimensions
        noise = torch.randn_like(image)

        timestep = self.timesteps[timestep]
        sigma = self.get_sigmas([timestep])

        # noise the images
        noised_image = (1.0 - sigma) * image + noise * sigma

        # returning noise is helpful for training
        return noised_image, noise
    
    @torch.no_grad()
    def reverse_flow(self, current_sample: torch.Tensor, model_output: torch.FloatTensor, timestep: float, stochasticity: bool):

        """ Function to integerate the reverse process (eval mode) for a latent by solving ODE by Euler's method """

        # when timestep is zero, the data has been reached
        if timestep <= 0:
            return current_sample
        
        # upcast to avoid precision errors
        current_sample = current_sample.to(torch.float32)

        # get the current and next sigma and the change between them
        current_sigma = self.sigmas[self.step_index]
        next_sigma = self.sigmas[self.step_index + 1]
        dt = next_sigma - current_sigma

        # stochasticity helps in randomizing what the model generates (increases diversity)
        if stochasticity:
            noise = torch.randn_like(current_sample)
            x_prev = current_sample - current_sigma * model_output
            prev_sample = (1 - next_sigma) * x_prev  + noise * next_sigma

        else:
            prev_sample = current_sample + dt * model_output

        self.step_index += 1

        return prev_sample
    
def test_noise():

    import os
    from PIL import Image
    from torchvision.transforms import ToTensor

    image_dir = os.path.join(os.getcwd(), "assets", "cat.png")
    image = Image.open(image_dir)
    image = ToTensor()(image)
    
    noiser = NoiseScheduler()
    noised_image, _ = noiser.add_noise(image = image.unsqueeze(0))
    
    import matplotlib.pyplot as plt
    print(noised_image.size())
    plt.figure()
    plt.imshow(noised_image.view(3, noised_image.size(-1), noised_image.size(-1)).permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    test_noise()