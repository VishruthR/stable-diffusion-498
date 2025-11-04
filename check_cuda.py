import torch

def main():
    print("PyTorch version:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)

    if cuda_available:
        print("CUDA Device Count:", torch.cuda.device_count())
        print("Current CUDA Device:", torch.cuda.current_device())
        print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
        # Simple tensor test
        x = torch.rand(3, 3).cuda()
        print("Tensor on CUDA:", x)
    else:
        print("CUDA not accessible. Check installation and drivers.")

if __name__ == "__main__":
    main()
