import torch
import torch.nn as nn
import sys
sys.path.append("/fs/pool/pool-marsot/")
import wandb
from datetime import datetime
from tankbind_philip.TankBind.tankbind.utils import *
from tqdm import tqdm
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
run = wandb.init(project="TankBind", name=f"{timestamp}")
from torch.utils.data import RandomSampler
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_dense_batch
from data import TankBindDataSet
from torch.nn import Linear
import sys
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from GINv2 import GIN
import xformers.ops as xops
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import to_dense_batch
import torch
from tankbind_philip.TankBind.tankbind.model import IaBNet_with_affinity


class TankBindDataLoader(torch.utils.data.DataLoader):
    """Subclass of the torch DataLoader, in order to apply the collate function TankBindCollater."""
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=None,
                 exclude_keys=None,
                 make_divisible_by_8=True,
                 **kwargs):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.make_divisible_by_8=make_divisible_by_8
        super().__init__(dataset,
                         batch_size,
                         shuffle,
                         collate_fn=TankBindCollater(dataset, follow_batch, exclude_keys, make_divisible_by_8=self.make_divisible_by_8),
                         **kwargs)



class TankBindCollater(Collater):
    """Applies batching operations and computations of masks in place of the model, in order to avoid having to recompute it in the
    forward pass on GPU."""
    def __init__(self, dataset,
                 follow_batch=None,
                 exclude_keys=None,
                 make_divisible_by_8=True):
        super().__init__(dataset, follow_batch, exclude_keys)
        self.make_divisible_by_8 = make_divisible_by_8
    def __call__(self, batch):
        data = super().__call__(batch)
        if self.make_divisible_by_8:
            max_dim_divisible_by_8_protein = 8 * (torch.diff(data["protein"].ptr).max() // 8 + 1)
            max_dim_divisible_by_8_compound = 8 * (torch.diff(data["compound"].ptr).max() // 8 + 1)
        else:
            max_dim_divisible_by_8_protein = torch.diff(data["protein"].ptr).max()
            max_dim_divisible_by_8_compound = torch.diff(data["compound"].ptr).max()
        protein_coordinates_batched, _ = to_dense_batch(
            data.node_xyz, data["protein"].batch,
            max_num_nodes=max_dim_divisible_by_8_protein,
            )
        protein_pairwise_representation = get_pair_dis_index(
            protein_coordinates_batched,
            bin_size=2,
            bin_min=-1,
            bin_max=protein_bin_max,
            ) # shape [batch_n, max_protein_size, max_protein_size, 16]
        _compound_lengths = (data["compound"].ptr[1:] - data["compound"].ptr[:-1]) ** 2
        _total = torch.cumsum(_compound_lengths, 0)
        compound_pairwise_distance_batch = torch.zeros(
                _total[-1], dtype=torch.long
            )
        for i in range(len(_total) - 1):
            compound_pairwise_distance_batch[_total[i] : _total[i + 1]] = i + 1
        compound_pair_batched, compound_pair_batched_mask = to_dense_batch(
            data.compound_pair,
            data.compound_pair_batch,
            )
        compound_pairwise_representation = torch.zeros(
            (len(batch), max_dim_divisible_by_8_compound, max_dim_divisible_by_8_compound, 16),
            dtype=torch.float32,
            )
        for i in range(len(batch)):
            one = compound_pair_batched[i]
            compound_size_square = (compound_pairwise_distance_batch == i).sum()
            compound_size = int(compound_size_square**0.5)
            compound_pairwise_representation[i, :compound_size, :compound_size] = one[
                :compound_size_square
                ].reshape((compound_size, compound_size, -1))
        data.batch_n = len(batch)
        data.max_dim_divisible_by_8_protein = max_dim_divisible_by_8_protein
        data.max_dim_divisible_by_8_compound = max_dim_divisible_by_8_compound
        data["protein", "p2p", "protein"].pairwise_representation = protein_pairwise_representation
        data["compound", "p2p", "compound"].pairwise_representation = compound_pairwise_representation
        data["compound", "p2p", "compound"].pairwise_representation_mask = compound_pair_batched_mask
        return data




def get_pair_dis_index(d, bin_size=2, bin_min=-1, bin_max=30):
    """
    Computing pairwise distances and binning.
    """
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    return pair_dis_bin_index

protein_bin_max = 30
def get_data(addNoise=None):
    pre = "./"
    add_noise_to_com = float(addNoise) if addNoise else None

    new_dataset = TankBindDataSet("/fs/pool/pool-marsot/pdbbind/pdbbind2020/dataset", add_noise_to_com=add_noise_to_com)
    new_dataset.data = new_dataset.data.query("c_length < 100 and native_num_contact > 5").reset_index(drop=True)
    d = new_dataset.data
    only_native_train_index = d.query("use_compound_com and group =='train'").index.values
    train = new_dataset[only_native_train_index]
    train_index = d.query("group =='train'").index.values
    train_after_warm_up = new_dataset[train_index]
    valid_index = d.query("use_compound_com and group =='valid'").index.values
    valid = new_dataset[valid_index]
    test_index = d.query("use_compound_com and group =='test'").index.values
    test = new_dataset[test_index]

    all_pocket_test_fileName = "/fs/pool/pool-marsot/tankbind_philip/TankBind/dataset/test_dataset"
    all_pocket_test = TankBindDataSet(all_pocket_test_fileName)
    all_pocket_test.compound_dict = "/fs/pool/pool-marsot/pdbbind/pdbbind2020/dataset/processed/compound.pt"
    info = None
    return train, train_after_warm_up, valid, test, all_pocket_test, info


def get_model(mode, logging, device):
    if mode == 0:
        logging.info("5 stack, readout2, pred dis map add self attention and GVP embed, compound model GIN")
        model = IaBNet_with_affinity().to(device)
    return model
device = torch.device("cuda:0")
model = IaBNet_with_affinity()
train, train_after_warm_up, valid, test, all_pocket_test, info = get_data(addNoise=5)
sampler = RandomSampler(train, replacement=True, num_samples=20000)
train_loader = TankBindDataLoader(train, batch_size=5, follow_batch=['x', 'compound_pair'], sampler = sampler)
sampler_2 = RandomSampler(train_after_warm_up, replacement=True, num_samples=20000)
train_after_warmup_loader = TankBindDataLoader(train_after_warm_up, batch_size=8, follow_batch=['x', 'compound_pair'], sampler = sampler_2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()
affinity_criterion = nn.MSELoss()

for epoch in range(200):
    model.train()
    model.to(device)
    y_list = []

    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    batch_loss = 0.0
    affinity_batch_loss = 0.0
    data_it = tqdm(train_after_warmup_loader)

    for data in data_it:
        data = data.to(device)
        optimizer.zero_grad()

        y_pred, affinity_pred = model(data)
        y = data.y
        affinity = data.affinity
        dis_map = data.dis_map
        y_pred = y_pred[data.equivalent_native_y_mask]
        y = y[data.equivalent_native_y_mask]
        dis_map = dis_map[data.equivalent_native_y_mask]


        contact_loss = criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0]).to(dis_map.device)
        y_pred = y_pred.sigmoid()


        relative_k = 0.01


        native_pocket_mask = data.is_equivalent_native_pocket
        affinity_loss =  relative_k * my_affinity_criterion(affinity_pred,
                                                            affinity, 
                                                            native_pocket_mask, decoy_gap=1.0)

        loss = contact_loss + affinity_loss
        wandb.log({"contact_loss":contact_loss.detach(), "affinity_loss":affinity_loss.detach(), "loss":loss.detach()})
        loss.backward()
        optimizer.step()
        batch_loss += len(y_pred)*contact_loss.item()
        affinity_batch_loss += len(affinity_pred)*affinity_loss.item()
        # print(f"{loss.item():.3}")
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        # torch.cuda.empty_cache()

    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)

    y_pred = torch.clip(1 - (y_pred / 10.0), min=1e-6, max=0.99999)
    contact_threshold = 0.2

    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)
    metrics = {"loss":batch_loss/len(y_pred) + affinity_batch_loss/len(affinity_pred)}