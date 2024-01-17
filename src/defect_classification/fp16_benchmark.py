from torch_snippets import Info
import torch, torch_tensorrt
from .basic_benchmark import benchmark, get_roc_auc_score

if __name__ == '__main__':
    model = torch.load('model.pth').eval()
    input_shape = (32,3,224,224)
    Info(f'Loading trt model...')
    trt_model_hp = torch_tensorrt.compile(
        model,
        inputs= [torch_tensorrt.Input(input_shape)],
        enabled_precisions= {torch_tensorrt.dtype.half} # Run with FP16
    )
    get_roc_auc_score(model)
    benchmark(trt_model_hp)