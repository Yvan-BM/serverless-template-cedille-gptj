from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
# import convert
# torch.multiprocessing.set_start_method('spawn', force=True)

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
# transformers.models.gptj.modeling_gptj.GPTJBlock = convert.GPTJBlock
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

def init():
    global model
    global tokenizer
    
    print("Tokenizer loading on cpu...")
    tokenizer = AutoTokenizer.from_pretrained("Cedille/fr-boris")
    print("done")

    print("Model loading on cpu...")
    model = transformers.AutoModelForCausalLM.from_pretrained("Cedille/fr-boris")
    print("done")
    
    if device == "cuda:0":
        print("Model passing on gpu...")
        model.to(device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    print("request tokenize")
    input_tokens = tokenizer(prompt, return_tensors='pt')
    input_tokens = {key: value.to(device) for key, value in input_tokens.items()}
    
    # Run the model
    print("request generate")
    output = model.generate(**input_tokens, min_length=18, max_length=20, do_sample=True)
    print("done")
    
    # Decode output token
    print("request decode")
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("done")
    
    result = {"output": output_text}

    # Return the results as a dictionary
    return result
