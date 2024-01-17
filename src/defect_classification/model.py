from torch_snippets import *

class SDD(nn.Module):
  classes = ['defect','non_defect']
  def __init__(self, model):
    super().__init__()
    self.model = model

  @torch.no_grad()
  def forward(self, x):
    x = x.view(-1,3,224,224).to(device)
    pred = self.model(x)
    conf = pred[0][0]
    clss = np.where(conf.item()<0.5,'non_defect','defect')
    print(clss)
    return clss.item()

  def predict(self, input):
    print(input)
    if isinstance(input, PIL.PngImagePlugin.PngImageFile):
      im = (255*np.array(input)).astype(np.uint8)
    else:
      im = (cv2.imread(input)[:,:,::-1])
    im = resize(im, 224)
    im = torch.tensor(im/255)
    im = im.permute(2,0,1).float()
    clss = self.forward(im)
    return {"class": clss}

