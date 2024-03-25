from torch_snippets import *
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
import time

from .train import process_example, DefectsDataset

def get_roc_auc_score(model):
    print("Started computing roc auc score...")
    predictions, actuals = [], []

    val_ds = load_dataset('sizhkhy/kolektor_sdd2', split="valid[:50]+valid[-50:]")
    val_ds = val_ds.map(process_example).remove_columns(['split', 'path'])
    val_ds.set_format("pt", columns=["image", "label"], output_all_columns=True)
    val_ds = DefectsDataset(val_ds)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, drop_last=True)

    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        if isinstance(model, nn.Module):
          prediction = model(x.cuda()).detach().cpu().numpy().tolist()
        else: # half/int8 model
          prediction = model(x.cuda())[0].detach().cpu().numpy().tolist()
        predictions.extend(prediction)
        actuals.extend(y.detach().cpu().numpy().tolist())

    actuals = flatten(actuals)
    predictions = flatten(predictions)
    print(f"ROC AUC Score: {roc_auc_score(actuals, predictions):.2f}")


@torch.no_grad()
def benchmark(model, input_shape=(32, 3, 32, 32), nwarmup=5, nruns=100):
    print("Started benchmarks...")
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    for _ in range(nwarmup):
        model(input_data)
    torch.cuda.synchronize()

    timings = []
    for _ in range(nruns):
        start_time = time.perf_counter()
        model(input_data)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    timing = np.mean(timings)*1000
    print(f'Average batch time: {timing:.2f} ms')


if __name__ == '__main__':
    model = torch.load('model.pth').cuda().eval()
    get_roc_auc_score(model)
    benchmark(model)
