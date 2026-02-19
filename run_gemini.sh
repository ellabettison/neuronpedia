
sudo make inference-localhost-install
# CUDA
sudo make inference-localhost-build-gpu USE_LOCAL_HF_CACHE=1
# CUDA
sudo make inference-localhost-dev-gpu \
     MODEL_SOURCESET=gemma-2-9b.gemmascope-res-16k \
     USE_LOCAL_HF_CACHE=1