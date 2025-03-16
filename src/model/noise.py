import torch
import torch.nn as nn
from typing import Optional

class NoiseScheduler(torch.nn.Module):
    # Add Noise to an image gradually
    def __init__(self, beta: int, timesteps: int, mu: float = 0.0, sigma: float = 1.0):
        super(NoiseScheduler, self).__init__()

        self.beta = beta
        self.mu = mu
        # control shape of logit-normal distrubition
        self.sigma = sigma

        self.timesteps = timesteps
        # compute alpha values so we can index into them later
        self.alphas = 1 - self.beta * torch.linspace(0, 1, timesteps)
        # alpha bar represents our mean
        # refrence equation: https://miro.medium.com/v2/resize:fit:242/format:webp/1*nIl1f0BYyqXM3TIDHHUaWg.png
        self.alpha_bar = torch.cumprod(self.alphas, dim = 0) #cumulative product

    def check_timestep(self, timestep: int):
        # check if timestep is valid
        if timestep > self.timesteps:
            raise ValueError("Can't have a timestep larger than the defined timestep")
        
    def sample_logit_timestep(self, batch_size: int = 1, device: str = "cpu"):
        " Sample timesteps from logit normal distribution "
        # returns timesteps in range [0, timesteps - 1]
        # beneficial to train the model on different noise levels

        normal_samples = torch.randn(batch_size, device = device) * self.sigma + self.mu

        sampled_t = nn.Sigmoid()(normal_samples)

        weighted_sampled_t = (sampled_t * (self.timesteps - 1)).long() # scale

        return weighted_sampled_t
        


    def add_noise(self, image: torch.Tensor,  timestep: int = None):

        """
        Adds noise to the image according to the timestep.
        t = Timestep at which to evaluate the noise schedule.
                           If None, uses the final timestep.
        """

        if timestep == None:
            timestep = self.sample_logit_timestep(image.size(0), device = image.device)

        self.check_timestep(timestep)

        # rand_like will use uniform random numbers to generate noise
        # that will fit the image's dimensions
        noise = torch.rand_like(image.float())
        
        noised_image = torch.sqrt(self.alpha_bar[timestep]) * image + noise * torch.sqrt(1 - self.alpha_bar[timestep])

        # returning noise is helpful for training
        return noised_image, noise
    
    def euler_solver(self, model, x: torch.Tensor, dt: Optional[float], steps: int = 20, stochasticity: bool = True):
        " Implements Euler's ODE solver "

        if not dt: dt = 1 / steps
        self.model = model

        for i in range(steps):
            current_time = 1 - i * dt  # Current time decreasing from 1 to 0
            xt = self.reverse_flow(x, dt, current_time, stochasticity = stochasticity)
        
        return xt.cpu()
    
    @torch.no_grad()
    def rk4_solver(self, model, x: torch.Tensor, dt: Optional[float] = None, steps: int = 20, stochasticity: bool = True):
        " 4th-order Runge-Kutta ODE solver "

        if not dt: dt = 1 / steps
        self.model = model

        for i in range(steps):
            # current time decreases from 1 to 0
            current_time = 1 - i  * dt

            # compute the RK4 increments
            # k1 = f(x, t)
            k1 = self.reverse_flow(x, current_time, stochasticity = stochasticity)
            # k2 = f(x + dt/2 * k1, t - dt/2)
            k2 = self.reverse_flow(x + (dt / 2) * k1, current_time - dt / 2, stochasticity = stochasticity)
            # k3 = f(x + dt/2 * k2, t - dt/2)
            k3 = self.reverse_flow(x + (dt / 2) * k2, current_time - dt / 2, stochasticity = stochasticity)
            # k4 = f(x + dt * k3, t - dt)
            k4 = self.reverse_flow(x + dt * k3, current_time - dt, stochasticity = stochasticity)

        dx = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x + dx

        return x
    
    def reverse_flow(self, x: torch.Tensor, dt: float, current_time: float, stochasticity: bool):

        # dt: timestep for integeration (a small positive number)
        # current_time: the time value in [0, 1], where 1 is pure noise and 0 is data

        # when current time is zero, the data has been reached
        if current_time <= 0:
            return x
        
        # velocity field v(x, t)
        with torch.no_grad():
            x_prev = self.model(x, current_time)
        
        # TODO: check if this is valid
        # stochasticity helps in randomizing what the model generates (increases diversity)
        if stochasticity:
            noise = torch.randn_like(x)
            # scale the noise by sqrt(dt) so variance is proportional to dt.
            x_prev = x_prev + (torch.sqrt(torch.tensor(dt, device=x.device)) * noise)

        return x_prev
    
    @property
    def get_alpha(self):
        return self.alpha_bar
    
def test_noise():

    import os
    import torchvision.io as io

    image_dir = os.path.join(os.getcwd(), "assets", "dog.png")
    image = io.read_image(image_dir)
    
    noiser = NoiseScheduler(0.9, 10)
    noised_image, _ = noiser.add_noise(image = image.unsqueeze(0))

    print(noised_image)
    # TODO: save image and check noise

test_noise()