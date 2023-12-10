import torch
from torch.multiprocessing import Pool, Process, set_start_method, freeze_support

device = torch.device("cuda:0")

def process_point(i):
    print('Index:', i)

if __name__ == '__main__':
    # Freeze support is recommended for Windows environments
    freeze_support()

    # Assuming uv is a PyTorch tensor
    uv = torch.tensor([[1, 2], [3, 4], [5, 6]], device=device)

    # Use multiple processes to parallelize the process_point function
    with Pool() as pool:
        pool.map(process_point, range(len(uv)))

    # Explicitly close the pool to ensure it terminates
    pool.close()
    pool.join()
