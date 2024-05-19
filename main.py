import os
import torch
import random
import json
import numpy as np
import pandas as pd
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from cleanlab.filter import find_label_issues
from sklearn.model_selection import KFold

import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

import pytorch_lightning as pl

from src.review_data_module import ReviewDataModule
from src.sentiment_classifier_system import SentimentClassifierSystem

train_data_dir = os.environ['SM_CHANNEL_TRAIN']

def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ckpt_dir', type=str, default="./log")
    parser.add_argument('--review_save_dir', type=str, default="./log")

    return parser.parse_known_args()

def to_json(x, filepath):
  with open(filepath, 'w') as fp:
    json.dump(x, fp)


def init_system(args):
    r"""Start node.
    Set random seeds for reproducibility, and 
    instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(args)
    
    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = args.ckpt_dir,
      monitor = 'avg_val_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(args, callbacks = [checkpoint_callback])

    trainer = Trainer(
      max_epochs = args.max_epochs,
      callbacks = [checkpoint_callback])
    
    return dm, system, trainer

def train_test(args, dm, system, trainer):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    trainer.fit(system, dm)
    trainer.test(system, dm, ckpt_path = 'best')

    # results are saved into the system
    results = system.test_results

    # print results to command line
    pprint(results)

    preresults_path = '/opt/ml/output/data/pre-results.json'
    to_json(results, preresults_path)
    
def crossval(args, dm):
    r"""Confidence learning requires cross-validation to compute 
    out-of-sample probabilities for every element. Each element
    will appear in a single cross-validation split exactly once. 
    """
    # combine training and dev datasets
    X = np.concatenate([
      np.asarray(dm.train_dataset.embedding.cpu()),
      np.asarray(dm.dev_dataset.embedding.cpu()),
      np.asarray(dm.test_dataset.embedding.cpu()),
    ])
    y = np.concatenate([
      np.asarray(dm.train_dataset.data.label),
      np.asarray(dm.dev_dataset.data.label),
      np.asarray(dm.test_dataset.data.label),
    ])

    probs = np.zeros(len(X))  # we will fill this in

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = KFold(n_splits=3)    # create kfold splits

    for train_index, test_index in kf.split(X):
      probs_ = None
      # ===============================================
      # Get train and test slices of X and y.
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # Convert to torch tensors.
      X_train_tensor = torch.tensor(X_train).float()
      y_train_tensor = torch.tensor(y_train).long()
      X_test_tensor = torch.tensor(X_test).float()
      y_test_tensor = torch.tensor(y_test).long()

      # Create train/test datasets using tensors.
      train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
      test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

      # Create train/test data loaders from datasets.
      dl_train = DataLoader(train_dataset, batch_size=args.batch_size)
      dl_test = DataLoader(test_dataset, batch_size=args.batch_size)

      # Create a ModelCheckpoint callback for saving the best model during training
      model_checkpoint = ModelCheckpoint(
          dirpath = args.ckpt_dir,
          monitor = 'val_loss',
          mode = 'min',    # look for lowest `val_loss`
          save_top_k = 1,  # save top 1 checkpoints
          verbose = True,
      )

      # Create `SentimentClassifierSystem`.
      system = SentimentClassifierSystem(args, callbacks=[model_checkpoint])
      trainer = Trainer(max_epochs=args.max_epochs, callbacks=[model_checkpoint])

      trainer.fit(system, dl_train)
      probs_ = trainer.predict(system, dl_test)
      
      probs_= np.concatenate([p.numpy() for p in probs_], axis=0)
      # ===============================================
      assert probs_ is not None, "`probs_` is not defined."
      
      probs[test_index] = probs_.reshape(-1)
      
    # create a single dataframe with all input features
    all_df = pd.concat([
      dm.train_dataset.data,
      dm.dev_dataset.data,
      dm.test_dataset.data,
    ])
    all_df = all_df.reset_index(drop=True)
    # add out-of-sample probabilities to the dataframe
    all_df['prob'] = probs

    # save to excel file
    all_df.to_csv('/opt/ml/output/data/prob.csv', index=False)
    
    return all_df



def inspect(all_df):
    r"""Use confidence learning over examples to identify labels that 
    likely have issues with the `cleanlab` tool. 
    """
    prob = np.asarray(all_df.prob)
    prob = np.stack([1 - prob, prob]).T
  
    # rank label indices by issues
    ranked_label_issues = find_label_issues(
        labels=all_df.label.values, 
        pred_probs=prob, 
        return_indices_ranked_by='self_confidence'
    )

    assert ranked_label_issues is not None, "`ranked_label_issues` not defined."

    # save this to class
    issues = ranked_label_issues
    print(f'{len(ranked_label_issues)} label issues found.')

    # overwrite label for all the entries in all_df
    for index in issues:
      label = all_df.loc[index, 'label']
      # we FLIP the label!
      all_df.loc[index, 'label'] = 1 - label
  
    return issues, all_df


def review(issues, all_df):
    r"""Format the data quality issues found such that they are ready to be 
    imported into LabelStudio. We expect the following format:

    [
      {
        "data": {
          "text": <review text>
        },
        "predictions": [
          {
            "value": {
              "choices": [
                  "Positive"
              ]
            },
            "from_name": "sentiment",
            "to_name": "text",
            "type": "choices"
          }
        ]
      }
    ]

    See https://labelstud.io/guide/predictions.html#Import-pre-annotations-for-text.and

    You do not need to complete anything in this function. However, look through the 
    code and make sure the operations and output make sense.
    """
    outputs = []
    for index in issues:
      row = all_df.iloc[index]
      output = {
        'data': {
          'text': str(row.review),
        },
        'predictions': [{
          'result': [
            {
              'value': {
                'choices': [
                  'Positive' if row.label == 1 else 'Negative'
                ]
              },
              'id': f'data-{index}',
              'from_name': 'sentiment',
              'to_name': 'text',
              'type': 'choices',
            },
          ],
        }],
      }
      outputs.append(output)

      # save to file
    preanno_path = '/opt/ml/output/data/pre-annotations.json'
    to_json(outputs, preanno_path)

def retrain_retest(args, all_df):
    r"""Retrain without reviewing. Let's assume all the labels that 
    confidence learning suggested to flip are indeed erroneous."""
    dm = ReviewDataModule(args)
    train_size = len(dm.train_dataset)
    dev_size = len(dm.dev_dataset)
    test_size = len(dm.test_dataset)

    # Calculate the indices for each dataset split based on their original sizes
    train_indices = range(train_size)
    dev_indices = range(train_size, train_size + dev_size)
    test_indices = range(train_size + dev_size, train_size + dev_size + test_size)

    # Update the dataframes in the datasets
    dm.train_dataset.data = all_df.iloc[train_indices].reset_index(drop=True)
    dm.dev_dataset.data = all_df.iloc[dev_indices].reset_index(drop=True)
    dm.test_dataset.data = all_df.iloc[test_indices].reset_index(drop=True)

    # Create a ModelCheckpoint callback for saving the best model during training
    model_checkpoint = ModelCheckpoint(
        monitor='avg_val_loss',
        dirpath='/opt/ml/model/',
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    system = SentimentClassifierSystem(args, callbacks=[model_checkpoint])
    trainer = Trainer(
      max_epochs=args.max_epochs, callbacks=[model_checkpoint])

    trainer.fit(system, dm)
    trainer.test(system, dm, ckpt_path='best')
    results = system.test_results

    pprint(results)
    
    finalresults_path = '/opt/ml/output/data/final-results.json'
    to_json(results, finalresults_path)

def start(args):
    
    # Step 1: Initialization
    dm, system, trainer = init_system(args)
    
    # Step2 : Training and offline evaluation
    train_test(args, dm, system, trainer)
    
    # Step 3: cross-validation
    all_df = crossval(args, dm)

    # Step 4: Inspection
    issues, all_df = inspect(all_df)
    
    # Step 5: Review
    review(issues, all_df)

    # Step 6: Re-train and Re-test
    retrain_retest(args, all_df)
    
    # Step 7: End
    """End node!"""
    print('done! great work!')
    

if __name__ == "__main__":
    
    # configuration files contain all hyperparameters
    args, _ = parse_args() 
    start(args)
