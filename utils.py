import os
import gc
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, concatenate_datasets
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay


def set_seed(seed):
    """
    Sets all random seeds.
    """
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # when running the cudnn backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"All random seeds set to {seed}.")
    return seed

def smoothen_list(list_to_smooth, n):
    """
    Smoothen list_to_smooth by averaging over the last up to n values.
    """
    smoothened_list = []
    for i in range(len(list_to_smooth)):
        if i<n:
            values_i = list_to_smooth[:i+1] # go from start until last postion with i<n
        else:
            values_i = list_to_smooth[i+1-n:i+1]
        smoothened_list.append(np.mean(values_i))
    return smoothened_list

def make_confusion_matrix(y_true, y_pred, labels, percentage=True, plot=True, size=7, output_dir=None):
    """
    This function makes a confusion matrix.
    parameters:
        - y_true: list of true labels
        - y_pred: list of predicted labels
        - labels: list of label names (=> list of strings) corresponding to y_true
        - percentage:
              - True (default) => the values in each row (that is for each actual class) a converted to percentages
              - False => keep raw counts
        - plot:
              - True (default) => plot confusion matrix
              - False => return confusion matrix as np.array
        - size: size of plot (7 by default)
        - folderpath: if specified, the plot will be saved as "confusion_matrix.png" under that folderpath
    """
    y_true_flat = y_true # flat list of labels
    y_pred_flat = y_pred # flat list of predictions
    idx_dict = {}
    dim = len(labels)
    for i in range(dim):
        idx_dict[labels[i]] = i # idx_dict["label(text)"] = index (indices to be used as rows or columns)
    conf_matrix = np.zeros((dim, dim)) # matrix with dim-by-dim zeroes
    for i in range(len(y_true_flat)): # use i to loop over both, actual and predicted labels
        row = y_true_flat[i] # true label
        col = y_pred_flat[i] # true label
        conf_matrix[row, col] += 1
    if percentage==True:
        values_format = ".1f"
        for i in range(len(conf_matrix)):
            conf_matrix[i] *= 100/np.sum(conf_matrix[i])
        cfm_type = "pct"
    else:
        values_format = "d"
        conf_matrix = conf_matrix.astype(int)
        cfm_type = "abs"
    if not plot:
        return conf_matrix
    cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(size, size))
    cmd.plot(ax=ax, xticks_rotation="vertical", values_format=values_format)
    plt.tight_layout()
    if output_dir!=None:
        output_filepath = f"{output_dir}/confusion_matrix_{cfm_type}.png"
        cmd.figure_.savefig(output_filepath)
        #print(f"confusion_matrix saved under path:\n{output_filepath}")
    pass

def get_max_instance(tokenized_dataset):
    """
    The tokenized_dataset is NOT a DatasetDict BUT a tokenized Dataset with an "input_ids" field.
    """
    assert isinstance(tokenized_dataset, Dataset), "passed dataset is not a `Dataset` instance"
    assert "input_ids" in list(tokenized_dataset.features.keys()), "passed dataset has no 'input_ids' field"
    index_max = -1
    len_max = -1
    for i in range(tokenized_dataset.num_rows):
        len_i = len(tokenized_dataset[i]["input_ids"]) # the length of "input_ids" (=> tokenized) may exceed the length of "words"
        if len_i > len_max:
            index_max = i
            len_max = len_i
    instance_max = tokenized_dataset[index_max]
    return instance_max, len_max

def get_max_batchsize(model, instance_max, data_collator):
    """
    Given a model and the biggest instance of a dataset, this function returns the biggest batch size for training the model.
    The passed model will remained untouched – instead, a copy of the model is used for finding the biggest batch size.
    Obviously, the result depends on the hardware.
    (bs is short for batch_size)
    """
    bs_accelerator = Accelerator()
    # try both:
    # 1. model = model
    # 2. model = copy.copy(model)
    optimizer = AdamW(model.parameters())
    # https://stackoverflow.com/questions/66266232/pandas-collapse-values-of-columns-to-lists
    df = pd.DataFrame.from_dict(instance_max)
    df = df.stack().reset_index(level=0, drop=True)
    df = df.groupby(df.index).apply(list).to_frame().transpose()
    # initialise parameters
    bs_max = 0
    model.train()
    #bs_datacollator = copy.copy(data_collator)
    stop_loop = False
    # start loop
    for bs_i in range(10*6):
        if stop_loop==False:
            bs_batch_size = 2**bs_i
            # build dataset of size "bs_batch_size"
            if bs_i==0:
                instance_max_ds = Dataset.from_pandas(df)
            else:
                instance_max_ds = concatenate_datasets([instance_max_ds, instance_max_ds])
            # define dataloader that returns the entire dataset of size "bs_batch_size"
            bs_dataloader = DataLoader(instance_max_ds, collate_fn=data_collator, batch_size=bs_batch_size)
            assert bs_batch_size==len(instance_max_ds)
            bs_batch = next(iter(bs_dataloader))
            try:
                bs_outputs = model(**bs_batch)
                bs_loss = bs_outputs.loss
                bs_accelerator.backward(bs_loss)
                optimizer.step()
                optimizer.zero_grad()
                bs_max = bs_batch_size
                print(f"Batch size\t{bs_max}\tworks!")
            except:
                del df, model#, bs_datacollator
                gc.collect()
                stop_loop = True
                break
    return bs_max
