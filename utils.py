import gc
import copy
import pandas as pd
from datasets import Dataset, concatenate_datasets
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

def get_max_instance(tokenized_dataset):
    """
    tokenized_dataset is NOT a DatasetDict BUT a tokenized Dataset with an "input_ids" field, 
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
    bs_model = copy.copy(model)
    optimizer = AdamW(bs_model.parameters())
    # https://stackoverflow.com/questions/66266232/pandas-collapse-values-of-columns-to-lists
    df = pd.DataFrame.from_dict(instance_max)
    df = df.stack().reset_index(level=0, drop=True)
    df = df.groupby(df.index).apply(list).to_frame().transpose()
    # initialise parameters
    bs_max = 0
    bs_model.train()
    bs_datacollator = copy.copy(data_collator)
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
            bs_dataloader = DataLoader(instance_max_ds, collate_fn=bs_datacollator, batch_size=bs_batch_size)
            assert bs_batch_size==len(instance_max_ds)
            bs_batch = next(iter(bs_dataloader))
            try:
                bs_outputs = bs_model(**bs_batch)
                bs_loss = bs_outputs.loss
                bs_accelerator.backward(bs_loss)
                optimizer.step()
                optimizer.zero_grad()
                bs_max = bs_batch_size
                print(f"batch size\t{bs_max}\tworks!")
            except:
                del df, bs_model, bs_datacollator
                gc.collect()
                stop_loop = True
                break
    return bs_max
