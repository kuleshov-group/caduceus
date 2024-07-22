# %%
from omegaconf import OmegaConf

import torch
from torch.utils.data.dataloader import DataLoader

import torch.backends

import pytorch_lightning as pl
from train import SequenceLightningModule

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders.genomics import TCGADataset

from caduceus.tokenization_caduceus import CaduceusTokenizer

import os

from tqdm import tqdm

import h5py

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#import pandas as pd
# %%
# OmegaConf.register_new_resolver('eval', eval)
# OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
# OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))


# %%
#config = "/mnt/bulk-neptune/timlenz/tumpe/caduceus/configs/experiment/tcga/tcga.yaml"
config = "/mnt/bulk-neptune/timlenz/tumpe/caduceus/outputs/2024-07-21/11-06-28-196644/config.json"
df_path = '/mnt/bulk-neptune/timlenz/tumpe/data/MUTATION/tcga_mutations_controlled.csv'
outdir = "/mnt/bulk-neptune/timlenz/tumpe/data/features/caduceus-mamba-2-1024-e6"
if not os.path.exists(outdir):
    os.makedirs(outdir)
# %%
config = OmegaConf.load(config)
#config["defaults"][1]["/model"]="caduceus"
config["defaults"]=[{'/pipeline': 'tcga'}, {'/model': 'caduceus'}, {'override /scheduler': 'cosine_warmup_timm'}]
#config["model"]={"config":{"d_model":512,"n_layer":16}}
#config["dataset"]["_name_"] = "tcga"
#config["dataset"]["df_path"] = df_path
#config["train"]["state"] = {"mode":None,"n_context":0,"n_context_eval":0}
#config["train"]["disable_dataset"] = False
config["train"]["pretrained_model_path"] = "/mnt/bulk-neptune/timlenz/tumpe/caduceus/outputs/2024-07-21/11-06-28-196644/checkpoints/last.ckpt"
config["dataset"]["mlm"]= False
#config["encoder"]=None
#config["decoder"]=None
#print(config)
config = utils.train.process_config(config)
utils.train.print_config(config, resolve=True)
model = SequenceLightningModule(config)

model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )
tokenizer = CaduceusTokenizer(
                model_max_length=config.dataset.max_length,
                add_special_tokens=False
            )
ds = TCGADataset("train",df_path,config.dataset.max_length,False, add_eos=config.dataset.add_eos,tokenizer=tokenizer)
# dl = DataLoader(
#             ds,
#             batch_size=1,
#             shuffle=False,
#             drop_last=False,)
model = model.cuda()

for i,pat in enumerate(tqdm(ds.pat_list)):
    data, _ = ds.__getitem__(i)
    data = data.cuda()
    with torch.no_grad():
        feats = model.model(data.unsqueeze(0),output_hidden_states=True)[1][16][:,-1][0]
        assert len(feats.shape)==1 and feats.shape[0]==512, f"{feats.shape=}"
    
    with h5py.File(f"{os.path.join(outdir,pat)}.h5","w") as f:
        f["feats"] = feats.detach().cpu().numpy()
# %%
