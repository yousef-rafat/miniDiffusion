import torch
import torch.nn as nn

class VAE(nn.Module):
    # Create the variational autoencoder
    def __init__(self, input_channels = 3, depth = 4, size = 14):
        """
        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space.
            depth (int): Number of downsampling (and upsampling) blocks.
        """
        # the goal of the code is to get the encoder and the decoder models with some helpful layers
        super(VAE, self).__init__()
        self.size = size

        # ////////////////////
        # Building the Encoder
        # ////////////////////

        encoder_layers = []
        in_channels = input_channels

        # Create a sample of conv blocks each downsampling by 2
        for i in range(depth):
            output_channels = 32 if i == 0 else in_channels * 2
            encoder_layers.append(
                nn.Conv2d(in_channels, output_channels, kernel_size = 3, stride = 2, padding = 1)
            )
            in_channels = output_channels

        # turn the layers into a model
        self.encoder_model = nn.Sequential(*encoder_layers)

        # conv layers for varianece and mean
        # use conv to keep the spaial dimensions
        self.fc_mu  = nn.Conv2d(in_channels, 4, kernel_size = 3, stride = 1, padding = 1) # mean
        self.fc_var = nn.Conv2d(in_channels, 4, kernel_size = 3, stride = 1, padding = 1) # variance

        # adaptive average pooling to allow any input size for the images
        self.avg_pool = nn.AdaptiveAvgPool2d((size, size)) # ensure fixed size before passing

        # later use for reshaping in decoding
        self.final_channels = in_channels

        # layer for decoding the latents
        #self.fc_decode = nn.Linear(4 * 28 * 28, 200704)
        self.fc_decode = nn.Linear(4 * size * size, self.final_channels * size * size)

        # ////////////////////
        # Building the Decoder
        # ///////////////////
        
        decoder_layers = []
        # latent dimension
        in_channels = self.final_channels
        # Create sample of up block each decreasing the depth by 2

        for i in range(depth):
            # reverse the channel doubling that happened in encoding
            output_channels = in_channels // 2
            decoder_layers.append(
                # stride = 2 for increasing the dimensions of the image
                nn.ConvTranspose2d(in_channels, output_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
            )
            # add batch norm and relu to better the vae decoder
            decoder_layers.append(nn.BatchNorm2d(output_channels))
            decoder_layers.append(nn.ReLU(inplace = True))

            in_channels = output_channels

        # last layer to get the image to the original number of channels
        decoder_layers.append(nn.Conv2d(in_channels, input_channels, kernel_size = 3))

        # we will add a sigmoid layer to ensure our output is between 0-1
        # without the sigmoid, the model will have to learn that the output is between 0 and 1
        
        decoder_layers.append(nn.Sigmoid())

        # get the decoder model
        self.decoder_model = nn.Sequential(*decoder_layers)

    def reparam_trick(self, mean: torch.Tensor, logvar: torch.Tensor):
        " Applied the reparamterization trick "
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)

        return mean + eps * std

    def encode(self, image: torch.Tensor):

        """
        Encodes the input image into a latent vector.
        Input Image = (batch, channels, height, width).
        """

        # function to encode image
        x = self.encoder_model(image.float())
        x = self.avg_pool(x) # get fixed size

        mean_u = self.fc_mu(x)
        logvar = self.fc_var(x)

        latent = self.reparam_trick(mean = mean_u, logvar = logvar)

        return mean_u, logvar, latent

    def decode(self, latent: torch.Tensor):
        " decodes a latent "

        # avoid flattening batch size
        x = torch.flatten(latent, start_dim = 1)

        x = self.fc_decode(x)

        batch_size = latent.size(0)

        x = x.reshape(batch_size, self.final_channels, self.size, self.size)

        image = self.decoder_model(x)

        return image
    

def load_vae(model: VAE, path: str, device: str = "cpu") -> VAE:
    # load checkpoint into the model

    checkpoint = torch.load(path, map_location = device)
    model = model.load_state_dict(checkpoint)

    model.eval()

    return model

def test_vae():

    vae = VAE()

    image = torch.randn(1, 3, 224, 224)

    _, _, latent = vae.encode(image)

    image = vae.decode(latent = latent)
    print(image.size())
    print("Image: ", image)
