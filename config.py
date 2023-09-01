import os
from vllm import SamplingParams, LLM

# Setting CUDA environment variables
CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"] = ''
RWKV_JIT_ON = os.environ["RWKV_JIT_ON"] = '1'
RWKV_CUDA_ON = os.environ["RWKV_CUDA_ON"] = '0'

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.7, max_tokens=256)

# Model name
model = LLM(model="egirlsai/egirls-chat")
