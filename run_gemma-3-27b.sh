
sudo make inference-localhost-install
# CUDA
sudo make inference-localhost-build-gpu USE_LOCAL_HF_CACHE=1
# CUDA
sudo make inference-localhost-dev-gpu \
     MODEL_SOURCESET=gemma-3-27b-it.gemmascope-2-res-65k \
     USE_LOCAL_HF_CACHE=1