import torch
import platform

def get_best_device(method_name=None):
    """
    Get the best available device for PyTorch operations.
    Priority: CUDA > MPS (Mac) > CPU
    
    Args:
        method_name: Optional method name to check for MPS compatibility
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        # Check if method has known MPS compatibility issues
        mps_incompatible_methods = ["mhi_baseline", "mhi_attention", "mhi_fusion"]
        if method_name and method_name in mps_incompatible_methods:
            device = "cpu"
            print("MPS detected but using CPU due to known compatibility issues with 3D operations")
        else:
            device = "mps"
            print("Using MPS (Metal Performance Shaders) device for Mac")
    else:
        device = "cpu"
        print("Using CPU device")
    
    return device

def to_device(tensor, device):
    """
    Move tensor to device with error handling for MPS
    """
    try:
        return tensor.to(device)
    except Exception as e:
        if device == "mps":
            print(f"MPS device error, falling back to CPU: {e}")
            return tensor.to("cpu")
        else:
            raise e

def device_info():
    """
    Print detailed device information
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available for Mac")
    
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
