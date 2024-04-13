# !pip install -U torch sentencepiece transformers accelerate

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BottleneckT5Autoencoder
#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ac = BottleneckT5Autoencoder(model_path='thesephist/contra-bottleneck-t5-large-wikipedia', device=device)

#%%
for t in range(10):
  print(ac.generate_from_latent(ac.embed('mom')//ac.embed('mother')))


# %%
ac.embed('vocoder/camouflage§')-ac.embed('camouflage/vocoder§§')