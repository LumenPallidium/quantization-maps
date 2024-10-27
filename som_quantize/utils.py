import torch
from einops.layers.torch import Rearrange
import numpy as np
from sympy.ntheory import factorint
from tqdm import tqdm

def losses_to_running_loss(losses, alpha = 0.95):
    running_losses = []
    running_loss = losses[0]
    for loss in losses:
        running_loss = (1 - alpha) * loss + alpha * running_loss
        running_losses.append(running_loss)
    return running_losses

def approximate_square_root(x):
    factor_dict = factorint(x)
    factors = []
    for key, item in factor_dict.items():
        factors += [key] * item
    factors = sorted(factors)

    a, b = 1, 1
    for factor in factors:
        if a <= b:
            a *= factor
        else:
            b *= factor
    return a, b

def tuple_checker(item, length):
    """Checks if an item is a tuple or list, if not, converts it to a list of length length.
    Also checks that an input tuple is the correct length.
    Useful for giving a function a single item when it requires a iterable."""
    if isinstance(item, (int, float, str)):
        item = [item] * length
    elif isinstance(item, (tuple, list)):
        assert len(item) == length, f"Expected tuple of length {length}, got {len(item)}"
    return item


#TODO : this class is unnecessary, keeping in case it's useful later
class Pairer:
    """Simple class using Cantor's pairing function or raster order to go from 2D to 1D and back.
    See https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.n = width * height

    def cantor_pair(self, x, y):
        assert x < self.width and y < self.height, "indices out of range"
        return int(((x + y) * (x + y + 1) / 2) + y)
    
    def cantor_unpair(self, z):
        assert z < self.n, "index is out of range"
        w = int(np.floor((np.sqrt(8 * z + 1) - 1) / 2))
        t = (w * w + w) / 2
        y = int(z - t)
        x = int(w - y)
        return x, y
    
    def raster_pair(self, x, y):
        assert x < self.width and y < self.height, "indices out of range"
        return int(x + y * self.width)
    
    def raster_unpair(self, z):
        assert z < self.n, "index is out of range"
        x = z % self.width
        y = z // self.width
        return x, y
    

class SOMGrid(torch.nn.Module):
    def __init__(self, 
                 height,
                 width,
                 neighbor_distance = 6,
                 kernel_type = "gaussian",
                 time_constant = 0.0005,
                 normalize = False,):
        super().__init__()
        self.height = height
        self.width = width
        self.size = width * height
        self.normalize = normalize

        self.t = 0
        self.time_constant = time_constant

        self.kernel_type = kernel_type
        self.neighbor_distance = neighbor_distance
        if neighbor_distance != 0:
            self.kernel_size = 2 * neighbor_distance + 1
            self.padding = neighbor_distance 
        else: # use the whole grid
            self.kernel_size = (2 * height + 1, 2 * width + 1)
            self.padding = (height, width)

        # layers to convert in and out of codebook
        self.codebook_to_grid = Rearrange("... dim (h w) -> ... dim h w", h = height, w = width)
        self.grid_to_codebook = Rearrange("... dim h w -> ... dim (h w)", h = height, w = width)

        if kernel_type == "gaussian":
            if isinstance(self.kernel_size, int):
                kernel_size_max = self.kernel_size
            else:
                kernel_size_max = max(self.kernel_size)
            range_ = kernel_size_max // 2
            kernel_init = torch.exp(-torch.arange(-range_, range_ + 1)**2)

            # use outer product to get 2d kernel
            kernel_init = torch.outer(kernel_init, kernel_init)

        elif kernel_type == "hard":
            if isinstance(self.kernel_size, int):
                kernel_size = (self.kernel_size, self.kernel_size)
            kernel_init = torch.ones(*kernel_size)
        #TODO : other kernel types may be interesting - eg triangular
        else:
            raise ValueError("kernel_type must be gaussian or hard")
        
        if self.normalize:
            kernel_init = kernel_init / kernel_init.sum()
        
        self.register_buffer("kernel_init", kernel_init)
        self.register_buffer("kernel", kernel_init)
        

    def update_t(self):
        self.t += 1
        t_scalar = torch.tensor(1 + self.t * self.time_constant)
        if self.kernel_type == "gaussian":
            kernel = (self.kernel_init).pow(t_scalar)
        else:
            kernel = self.kernel_init * (1 / t_scalar)
        
        if self.normalize:
            kernel = kernel / kernel.sum()

        self.kernel = kernel

    def forward(self, cb_onehot, update_t = True):
        _, dim, _ = cb_onehot.shape
        cb_reshaped = self.codebook_to_grid(cb_onehot)
        kernel = self.kernel[None, None, ...].repeat(dim, 1, 1, 1)
        cb_blurred = torch.nn.functional.conv2d(cb_reshaped, 
                                                kernel, 
                                                padding = self.padding,
                                                groups = dim)
        new_cb = self.grid_to_codebook(cb_blurred)

        if update_t:
            self.update_t()
        
        return new_cb

def quick_test(quantizer, n_clusters = 10, n_subclusters = 10, batch_size = 256, n_steps = 2000):
    """Quick visual test on a quantizer clustering points."""
    # import here to avoid dependency issues
    import matplotlib.pyplot as plt

    means = torch.randn(n_clusters, 2) * 20
    subcluster_means = torch.randn(n_subclusters, 2) * 4

    quantizer.train()

    for i in tqdm(range(n_steps)):
        # generate data
        data = 0.1 * torch.randn(batch_size, 2)
        data += means[torch.randint(n_clusters, (batch_size,))]
        data += subcluster_means[torch.randint(n_subclusters, (batch_size,))]

        _, _, _ = quantizer(data.unsqueeze(0))

    # get the codebook
    codebook = quantizer.get_codebook()
    plt.scatter(means[:, 0], means[:, 1],
                color = "blue", label = "clusters",
                marker = "*")
    combined_cluster = means[None, :, :] + subcluster_means[:, None, :]
    plt.scatter(combined_cluster[:, :, 0].flatten(),
                combined_cluster[:, :, 1].flatten(),
                color = "green", alpha = 0.5, label = "subclusters",
                marker = "x")
    if isinstance(codebook, list):
        plt.scatter(codebook[0][:, 0], codebook[0][:, 1],
                    color = "red", label = "codebook")
        combined_codebook = codebook[0][None, :, :] + codebook[1][:, None, :]
        plt.scatter(combined_codebook[:, :, 0].flatten(),
                    combined_codebook[:, :, 1].flatten(),
                    color = "purple", alpha = 0.5, label = "2nd CB")
    else:
        plt.scatter(codebook[:, 0], codebook[:, 1],
                    color = "red", label = "codebook")
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    from quantizer import ResidualQuantizer
    clusters = 10
    subclusters = 3
    n_iters = 2000
    quantizer = ResidualQuantizer(2, 2,
                                  clusters,
                                  probabilistic = True,
                                  temp = 1,
                                  use_som = False)
    quick_test(quantizer,
               n_clusters = clusters,
               n_subclusters = subclusters,
               n_steps = n_iters)



