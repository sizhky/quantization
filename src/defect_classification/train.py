from torch_snippets import *
from datasets import load_dataset, DatasetDict
from collections import Counter
from torchvision import models

class DefectsDataset:
    def __init__(self, ds):
        self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, ix):
        item = self.ds[ix]
        return item['image'], item['label']

def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        # nn.Linear(512, 32),
        nn.Linear(512, 128),
        nn.ReLU(),
        # nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    # loss_fn = nn.BCELoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()

def get_dataloaders(trn_ds, val_ds):
    train = DefectsDataset(trn_ds)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val = DefectsDataset(val_ds)
    val_dl = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
    return trn_dl, val_dl

def process_example(input):
  o = AD()
  im = input['image'].resize((224, 224))
  im = torch.Tensor(np.array(im)/255.).permute(2,0,1).float().cuda()
  lbl = torch.Tensor([int(input['label'] == 'defect')]).cuda()
  o.image = im
  o.label = lbl
  return o.to_dict()

def get_datasets(DEBUG):
    if DEBUG:
        trn_ds = load_dataset('sizhkhy/kolektor_sdd2', split="train[:50]+train[-50:]")
        val_ds = load_dataset('sizhkhy/kolektor_sdd2', split="valid[:50]+valid[-50:]")
        dataset = DatasetDict({'train': trn_ds, 'valid': val_ds})
    else:
        dataset = load_dataset('sizhkhy/kolektor_sdd2')
        trn_ds = dataset['train']
        val_ds = dataset['valid']

    print('Class Balance\n', AD(
        train=Counter(trn_ds['label']),
        valid=Counter(val_ds['label'])
    ))

    ds = dataset.map(process_example).remove_columns(['split'])
    ds.set_format("pt", columns=["image", "label"], output_all_columns=True)
    trn_ds = ds['train']
    val_ds = ds['valid']
    return trn_ds, val_ds

def main(DEBUG=True):
    model, criterion, optimizer = get_model()
    trn_ds, val_ds = get_datasets(DEBUG)
    trn_dl, val_dl = get_dataloaders(trn_ds, val_ds)
    model, loss_fn, optimizer = get_model()

    for epoch in (tracker:=track2(range(30))):
        train_epoch_losses = []
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch(x.cuda(), y.cuda(), model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
        if epoch % 10 == 0: print(f'Epoch: {epoch+1} {train_epoch_loss=:.3f}')
    torch.save(model, 'model.pth')
    os.symlink('model.pth', 'sdd.weights.pth')
    print(f"Saved model to model.pth")


if __name__ == '__main__':
    main()