import torch
from typing import Optional

class NoiseScheduler(torch.nn.Module):
    # Add Noise to an image gradually
    def __init__(self, model, beta: int, timesteps: int):
        super(NoiseScheduler, self).__init__()
        self.beta = beta
        self.timesteps = timesteps
        # compute alpha values so we can index into them later
        self.alphas = 1 - self.beta * torch.linspace(0, 1, timesteps)
        # alpha bar represents our mean
        # refrence equation: https://miro.medium.com/v2/resize:fit:242/format:webp/1*nIl1f0BYyqXM3TIDHHUaWg.png
        self.alpha_bar = torch.cumprod(self.alphas, dim = 0) #cumulative product

        self.model = model

    def check_timestep(self, timestep: int):
        # if no timestep was specified choose the one specified in initalization
        if timestep is None:
            timestep = self.timesteps
        elif timestep > self.timesteps:
            raise ValueError("Can't have a timestep larger than the defined timestep")


    def add_noise(self, image: torch.Tensor,  timestep: int = None):

        """
        Adds noise to the image according to the timestep.
        t = Timestep at which to evaluate the noise schedule.
                           If None, uses the final timestep.
        """

        self.check_timestep(timestep)

        # rand_like will use uniform random numbers to generate noise
        # that will fit the image's dimensions
        noise = torch.rand_like(image)
        
        output_tensor = torch.sqrt(self.alpha_bar[timestep]) * image + noise * torch.sqrt(1 - self.alpha_bar[timestep])

        # returning noise is helpful for training
        return output_tensor, noise
    
    def euler_solver(self, x: torch.Tensor, dt: Optional[float], steps: int = 20):
        " Implements Euler's ODE solver "

        if not dt: dt = 1 / steps

        for i in range(steps):
            current_time = 1 - i * dt  # Current time decreasing from 1 to 0
            xt = self.reverse_flow(x, dt, current_time)
        
        return xt.cpu()
    
    def reverse_flow(self, x: torch.Tensor, dt: float, current_time: float, stochasticity: bool = True):

        # dt: timestep for integeration (a small positive number)
        # current_time: the time value in [0, 1], where 1 is pure noise and 0 is data

        # when current time is zero, the data has been reached
        if current_time <= 0:
            return x
        
        # velocity field v(x, t)
        x_prev = self.model(x, current_time)
        
        # stochasticity helps in randomizing what the model generates (increases diversity)
        if stochasticity:
            noise = torch.randn_like(x)
            # scale the noise by sqrt(dt) so variance is proportional to dt.
            x_prev = x_prev + (torch.sqrt(torch.tensor(dt, device=x.device)) * noise)

        return x_prev
