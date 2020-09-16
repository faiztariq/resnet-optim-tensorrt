# Optimizing ResNet50 Model using TensorRT

Training a ResNet50 model on CIFAR10, optimizing the pytorch model (converting to ONNX) and running inference on NVIDIA GPU.

* Train a model
* Convert to ONNX
* Build a CUDA Engine using TensorRT APIs
* Run Inference

# A. Building & training PyTorch model
(./scripts/cifar_img_classification_with_inf_metrics.ipynb)

* Get the CIFAR10 dataset on your Google Drive.
* Mount the Drive on to Colab.
* Run the cells. (and update the paths wherever required)

# B. Converting the model from .pth to .onnx

* Run the script. (./scripts/model_opt.py)
* Save it in your Drive for running inference.

# C. Building Engine & Running Inference
(./inference/onnx_to_tensorrt_engine_with_inference_metrics.ipynb)

* Install the package and dependencies.
* Update the paths wherever required.
* Create inference set and run your inference.
