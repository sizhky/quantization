import torch, torch_tensorrt
from .basic_benchmark import benchmark, get_roc_auc_score
from .train import load_dataset, process_example, DefectsDataset, DataLoader

if __name__ == '__main__':
   model = torch.load('model.pth').eval()
   input_shape = (32,3,224,224)

   val_ds = load_dataset('sizhkhy/kolektor_sdd2', split="valid[:50]+valid[-50:]")
   val_ds = val_ds.map(process_example).remove_columns(['split', 'path'])
   val_ds.set_format("pt", columns=["image", "label"], output_all_columns=True)
   val_ds = DefectsDataset(val_ds)
   val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, drop_last=True)
   
   calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
      val_dl,
      use_cache=False,
      algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
      device=torch.device('cuda:0')
   )

   compile_spec = {
      "inputs": [torch_tensorrt.Input([32, 3, 224, 224])],
      "enabled_precisions": torch.int8,
      "calibrator": calibrator,
      "truncate_long_and_double": True
   }

   trt_ptq = torch_tensorrt.compile(model, **compile_spec)
   get_roc_auc_score(trt_ptq)
   benchmark(trt_ptq, input_shape=(32,3,224,224), nwarmup=50, nruns=100)
