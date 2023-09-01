import copy
import gc

from config import model, sampling_params, CUDA_VISIBLE_DEVICES, RWKV_JIT_ON, RWKV_CUDA_ON
CUDA_VISIBLE_DEVICES
RWKV_JIT_ON
RWKV_CUDA_ON

async def get_model():
    return model

async def evaluate(
    prompt,
    model
):
    return model.generate(prompt, sampling_params)

def remove_tokens(text):
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")

async def build_prompt(system_prompt, message, conversation):
    fullprompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
    
    for m in conversation:
        if m['role'] == 'user':
            fullprompt += f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            fullprompt += "<|im_start|>user\n" + remove_tokens(m['content']).strip() + "<|im_end|>\n"
        elif m['role'] == 'assistant':
            fullprompt += "<|im_start|>assistant\n" + remove_tokens(m['content']).strip() + "<|im_end|>\n"
            
    
    # trim message
    message = message.strip()
            
    fullprompt += "<|im_start|>user\n" + remove_tokens(message) + "<|im_end|>\n<|im_start|>assistant\n"
    print ("## Prompt ##")
    print (fullprompt)
    
    return fullprompt
    

async def handle_llama(system_prompt, message, conversation, model):
    
    system_prompt = system_prompt.strip() if system_prompt != None else ''
    
    fullprompt = await build_prompt(system_prompt, message, conversation, model)
    
    full_response = fullprompt
    for token, statee in evaluate(fullprompt, model):
        full_response += token
        yield token
        
    gc.collect()