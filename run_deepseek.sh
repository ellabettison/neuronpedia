
sudo make inference-localhost-install
# CUDA
sudo make inference-localhost-build-gpu USE_LOCAL_HF_CACHE=1
# CUDA
sudo make inference-localhost-dev-gpu \
     MODEL_SOURCESET=deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k \
     USE_LOCAL_HF_CACHE=1