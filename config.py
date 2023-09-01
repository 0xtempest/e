import os
from vllm import SamplingParams, LLM

from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "YOUR_REPO_ID"
FILENAME = "sklearn_model.joblib"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)

# Setting CUDA environment variables
CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"] = ''
RWKV_JIT_ON = os.environ["RWKV_JIT_ON"] = '1'
RWKV_CUDA_ON = os.environ["RWKV_CUDA_ON"] = '0'

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.7, max_tokens=256)

# Model name
model = LLM(model="egirlsai/egirls-chat")
