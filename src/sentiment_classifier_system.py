r"""A PyTorch Lightning system for training a sentiment classifier."""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pytorch_lightning as pl

class SentimentClassifierSystem(pl.LightningModule):
  """A Pytorch Lightning system to train a model to classify sentiment of 
  product reviews. 

  Arguments
  ---------
  args: input parameters
  """
  def __init__(self, args, callbacks):
    super().__init__()
    self.save_hyperparameters(args)

    # load model
    self.model = self.get_model()

    # We will overwrite this once we run `test()`
    self.test_results = {}
    
    self.model_checkpoint = callbacks[0]
    self.validation_step_outputs = []
    self.test_step_outputs = []

  def get_model(self):
    model = nn.Sequential(
      nn.Linear(768, self.hparams.model_width),
      nn.ReLU(),
      nn.Linear(self.hparams.model_width, 1),
    )
    return model
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), 
      lr=self.hparams.lr)
    return optimizer

  def _common_step(self, batch, _):
    """
    Arguments
    ---------
    embs (torch.Tensor): embeddings of review text
      shape: batch_size x 768
    labels (torch.LongTensor): binary labels (0 or 1)
      shape: batch_size
    """
    embs, labels = batch

    # forward pass using the model
    logits = self.model(embs)

    loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())

    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.round(torch.sigmoid(logits))
      num_correct = torch.sum(preds.squeeze() == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy

  def training_step(self, train_batch, batch_idx):
    loss, acc = self._common_step(train_batch, batch_idx)
    self.log_dict({'train_loss': loss, 'train_acc': acc},
      on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, dev_batch, batch_idx):
    loss, acc = self._common_step(dev_batch, batch_idx)
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
    return loss, acc

  def on_validation_epoch_end(self):
    avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
    avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
    self.log_dict({'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc},
      on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.validation_step_outputs.clear()

  def test_step(self, test_batch, batch_idx):
    loss, acc = self._common_step(test_batch, batch_idx)
    self.test_step_outputs.append({'test_loss': loss, 'test_acc': acc})
    return loss, acc

  def on_test_epoch_end(self):
    avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
    avg_acc = torch.stack([x['test_acc'] for x in self.test_step_outputs]).mean()
    results = {'loss': avg_loss.item(), 'acc': avg_acc.item()}
    self.test_results = results
    self.test_step_outputs.clear()

  def predict_step(self, batch, _):
    logits = self.model(batch[0])
    probs = torch.sigmoid(logits)
    return probs


