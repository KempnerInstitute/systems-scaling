import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.quantization import quantize_dynamic
import os

# Set the model name
model_name = "Qwen/Qwen2-0.5B"

# Load the model in full precision (FP32)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# Example usage
if __name__ == "__main__":
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nModel Parameter Data Types in Attention and MLP layers:")
    param_types = {}
    for name, param in model.named_parameters():
        if param is not None:
            print(f"{name}: {param.dtype}")
            param_types[param.dtype] = param_types.get(param.dtype, 0) + 1

    
    
    # Quantize the entire model dynamically
    # This properly handles the model structure without breaking it
    model = quantize_dynamic(
        model, 
        {nn.Linear},  # Only quantize linear layers
        dtype=torch.qint8,
    )

    # Move model to device first
    model = model.to(device)
    
    # Prepare dummy input (ensure it's bf16 and on same device)
    # Get vocab size from the model's configuration
    vocab_size = model.config.vocab_size
    batch_size = 1
    seq_len = 50
    
    # Prepare dummy input with correct dimensions and device
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    model.eval()
    # Print the model's data type
    print(f"Model dtype: {model.dtype}")
    
    # Print data type of model parameters focusing on attention and MLP layers
    print("\nModel Parameter Data Types in Attention and MLP layers:")
    param_types = {}
    for name, param in model.named_parameters():
        if param is not None:
            print(f"{name}: {param.dtype}")
            param_types[param.dtype] = param_types.get(param.dtype, 0) + 1
    
    # Print summary of all parameter types
    print("\nParameter Type Summary:")
    for dtype, count in param_types.items():
        print(f"{dtype}: {count} parameters")
    with torch.no_grad():
        # Generate predictions
        outputs = model.generate(inputs)
        # Print the datatype of outputs
        print(f"Outputs dtype: {type(outputs)}")
        print(f"Outputs object: {outputs}")

        # If outputs has attributes, examine their types too
        if hasattr(outputs, "__dict__"):
            print("\nOutputs attributes:")
            for attr_name in dir(outputs):
                if not attr_name.startswith("_") and not callable(getattr(outputs, attr_name)):
                    attr = getattr(outputs, attr_name)
                    print(f"  {attr_name}: {type(attr)}")
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

    loss_fn = nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.size(-1))           # reshape if needed
    targets = targets.view(-1)

    logits = logits.to(torch.float32)
    targets = targets.to(torch.long)
    loss = loss_fn(logits, targets)


    print(f"Loss: {loss.item()}")