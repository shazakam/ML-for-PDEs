import torch
import random

def generate_squares(initial_u: torch.Tensor):
    """Generate random squares with cyclic boundary conditions.

    Draws the borders of randomly positioned squares onto the tensor,
    wrapping around edges using modulo indexing.

    :param initial_u: 2D tensor to draw squares onto.
    :type initial_u: torch.Tensor
    :returns: Modified tensor with squares drawn.
    :rtype: torch.Tensor
    """
    num_squares: int = random.randint(1, 100)
    h, w = initial_u.shape

    for _ in range(num_squares):
        rand_x = random.randint(0, w - 1)
        rand_y = random.randint(0, h - 1)
        rand_size = random.randint(0, 100)
        rand_intensity = random.random() * random.randint(1, 10)

        rows = [(rand_y + i) % h for i in range(rand_size + 1)]
        cols = [(rand_x + i) % w for i in range(rand_size + 1)]

        # Top and bottom sides
        for col in cols:
            for t in range(-50, 50):
                initial_u[rows[0], (col + t) % w] = rand_intensity
                initial_u[rows[-1], (col + t) % w] = rand_intensity

        # Left and right sides
        for row in rows:
            for t in range(-50, 50):
                initial_u[(row + t) % h, cols[0]] = rand_intensity
                initial_u[(row + t) % h, cols[-1]] = rand_intensity

    return initial_u

def generate_normals(initial_u: torch.Tensor):
    """Add random 2D Gaussian distributions to the tensor.

    Superimposes between 1 and 8 axis-aligned Gaussian blobs onto ``initial_u``,
    each with a random centre, standard deviations, and peak intensity.

    :param initial_u: 2D tensor to add Gaussian distributions onto.
    :type initial_u: torch.Tensor
    :returns: Modified tensor with Gaussians added (in-place).
    :rtype: torch.Tensor
    """
    normal_gen = lambda x, y, A, x_0, y_0, sigma_x, sigma_y: A*torch.exp(-(0.5*((x-x_0)**2)/(sigma_x**2) + 0.5*((y-y_0)**2)/(sigma_y**2)))

    num_distributions = random.randint(1, 8)
    h, w = initial_u.shape

    for _ in range(num_distributions):
        canvas = torch.cartesian_prod(torch.tensor([i for i in range(h)]), torch.tensor([i for i in range(w)])).reshape((h, w, 2))
        rand_centre_x, rand_centre_y = random.randint(0, w-1), random.randint(0, h-1)
        rand_x_sigma, rand_y_sigma = random.randint(5,40), random.randint(5,40)
        rand_intensity = random.random()*random.randint(1,10)

        add_to_u = normal_gen(canvas[:,:,1], canvas[:,:,0], rand_intensity, rand_centre_x, rand_centre_y, rand_x_sigma, rand_y_sigma)

        initial_u +=  add_to_u

    return initial_u

def generate_paths(initial_u : torch.Tensor):

    num_paths = random.randint(200, 1200)
    h, w = initial_u.shape

    rand_intensity = [random.gauss(1, 0.2) for _ in range(num_paths)]
    rand_x_start = [random.randint(0, w-1) for _ in range(num_paths)]
    rand_y_start = [random.randint(0, h-1) for _ in range(num_paths)]
    path_lengths = [random.randint(50, 3*w) for _ in range(num_paths)]

    for idx, i in enumerate(range(num_paths)):
        start_loc = (rand_x_start[idx], rand_y_start[idx])
        path_intensity = rand_intensity[idx]
        path_length = path_lengths[idx]
        initial_u = generate_random_path(initial_u, start_loc, path_intensity, path_length)

    return initial_u

def generate_random_path(u : torch.Tensor, start_loc : tuple, path_intensity : float, path_length : int) -> torch.Tensor:
    x, y = start_loc
    h, w = u.shape
    u[x, y] = u[x,y] + path_intensity

    for _ in range(path_length):
        x_dif = random.choice([-1,1])
        y_dif = random.choice([-1,1])

        x, y = (x+x_dif)%w, (y+y_dif)%h

        u[x,y] += path_intensity

    return u
