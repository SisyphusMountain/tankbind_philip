import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, InMemoryDataset, download_url
from utils import construct_data_from_graph_gvp

class TankBind_prediction(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                pocket_radius=20, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        self.proteinMode = proteinMode
        self.pocket_radius = pocket_radius
        self.compoundMode = compoundMode
        self.shake_nodes = shake_nodes
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        pocket_com = line['pocket_com']
        pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
        pocket_com = pocket_com.reshape((1, 3))
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False

        protein_name = line['protein_name']
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]

        compound_name = line['compound_name']
        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = self.compound_dict[compound_name]

        # y is distance map, instead of contact map.
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                              protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                              use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)

        return data

class TankBindDataSet(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                add_noise_to_com=None, pocket_radius=20, contactCutoff=8.0, predDis=True, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None, use_esm_embeddings=False):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        if use_esm_embeddings:
            self.esm_embeddings = torch.load(self.processed_paths[3])
        self.add_noise_to_com = add_noise_to_com
        self.proteinMode = proteinMode
        self.compoundMode = compoundMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.shake_nodes = shake_nodes
        self.use_esm_embeddings = use_esm_embeddings
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt', 'esm_embeddings.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])
        torch.save(self.esm_embeddings, self.processed_paths[3])
    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        pocket_com = line['pocket_com']
        use_compound_com = line['use_compound_com']
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False
        group = line['group'] if "group" in line.index else 'train'
        add_noise_to_com = self.add_noise_to_com if group == 'train' else None

        protein_name = line['protein_name']
        if self.proteinMode == 0:
            protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]

        name = line['compound_name']
        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = self.compound_dict[name]

        shake_nodes = self.shake_nodes if group == 'train' else None
        if shake_nodes is not None:
            protein_node_xyz = protein_node_xyz + shake_nodes * (2 * np.random.rand(*protein_node_xyz.shape) - 1)
            coords = coords  + shake_nodes * (2 * np.random.rand(*coords.shape) - 1)

        if self.proteinMode == 0:
            if self.use_esm_embeddings:
                esm_embedding = self.esm_embeddings[protein_name]
            else:
                esm_embedding = None
            data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                  protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                  coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, contactCutoff=self.contactCutoff, includeDisMap=self.predDis,
                          pocket_radius=self.pocket_radius, add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, 
                          use_compound_com_as_pocket=use_compound_com, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode, esm_embedding=esm_embedding)

        affinity = float(line['affinity'])
        data.affinity = torch.tensor([affinity], dtype=torch.float)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)
        data.pdb = line['pdb'] if "pdb" in line.index else f'smiles_{idx}'
        data.group = group

        data.real_affinity_mask = torch.tensor([use_compound_com], dtype=torch.bool)
        data.real_y_mask = torch.ones(data.y.shape).bool() if use_compound_com else torch.zeros(data.y.shape).bool()

        if "native_num_contact" in line.index:
            fract_of_native_contact = (data.y.numpy() > 0).sum() / float(line['native_num_contact'])
            is_equivalent_native_pocket = fract_of_native_contact >= 0.9
            data.is_equivalent_native_pocket = torch.tensor([is_equivalent_native_pocket], dtype=torch.bool)
            data.equivalent_native_y_mask = torch.ones(data.y.shape).bool() if is_equivalent_native_pocket else torch.zeros(data.y.shape).bool()
        else:
            if "ligand_com" in line.index:
                ligand_com = line["ligand_com"]
                pocket_com = data.node_xyz.numpy().mean(axis=0)
                dis = np.sqrt(((ligand_com - pocket_com)**2).sum())
                # is equivalent native pocket if ligand com is less than 8 A from pocket com.
                is_equivalent_native_pocket = dis < 8
                data.is_equivalent_native_pocket = torch.tensor([is_equivalent_native_pocket], dtype=torch.bool)
                data.equivalent_native_y_mask = torch.ones(data.y.shape).bool() if is_equivalent_native_pocket else torch.zeros(data.y.shape).bool()
            else:
                # data.is_equivalent_native_pocket and data.equivalent_native_y_mask will not be available.
                pass
        return data


def get_data(data_mode="0", logging=None, addNoise=None, num_examples=None, use_esm_embeddings=False):
    pre = "./"
    if data_mode == "0":
        # logging.info(f"re-docking, using dataset: apr22_pdbbind_gvp_pocket_radius20 pred distance map.")
        # logging.info(f"compound feature based on torchdrug")
        add_noise_to_com = float(addNoise) if addNoise else None

        # compoundMode = 1 is for GIN model.
        #new_dataset = TankBindDataSet(f"{pre}/apr22_pdbbind_gvp_pocket_radius20", add_noise_to_com=add_noise_to_com)'
        new_dataset = TankBindDataSet("/fs/pool/pool-marsot/pdbbind/pdbbind2020/dataset", add_noise_to_com=add_noise_to_com, use_esm_embeddings=use_esm_embeddings)
        # modified by Enzo
        # load compound features extracted using torchdrug.
        # new_dataset.compound_dict = torch.load(f"{pre}/compound_dict.pt")
        new_dataset.data = new_dataset.data.query("c_length < 100 and native_num_contact > 5").reset_index(drop=True)
        d = new_dataset.data
        only_native_train_index = d.query("use_compound_com and group =='train'").index.values
        train = new_dataset[only_native_train_index]
        train_index = d.query("group =='train'").index.values
        if num_examples is not None:
            train_names = d.query("group == 'train'")["protein_name"].unique()
            rng = np.random.default_rng(12345)
            train_names = rng.choice(train_names, num_examples)
            train_index = d.query("group =='train' and protein_name in @train_names").index.values
        train_after_warm_up = new_dataset[train_index]
        # train = torch.utils.data.ConcatDataset([train1, train2])
        valid_index = d.query("use_compound_com and group =='valid'").index.values
        valid = new_dataset[valid_index]
        test_index = d.query("use_compound_com and group =='test'").index.values
        test = new_dataset[test_index]

        #all_pocket_test_fileName = f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20/"
        # added by Enzo
        all_pocket_test_fileName = "/fs/pool/pool-marsot/tankbind_philip/TankBind/dataset/test_dataset"
        all_pocket_test = TankBindDataSet(all_pocket_test_fileName)
        #all_pocket_test.compound_dict = torch.load(f"{pre}/compound_dict.pt")
        # added by Enzo
        all_pocket_test.compound_dict = "/fs/pool/pool-marsot/pdbbind/pdbbind2020/dataset/processed/compound.pt"
        # info is used to evaluate the test set. 
        info = None
        # info = pd.read_csv(f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)
    if data_mode == "1":
        # logging.info(f"self-docking, same as data mode 0 except using LAS_distance constraint masked compound pair distance")
        add_noise_to_com = float(addNoise) if addNoise else None

        # compoundMode = 1 is for GIN model.
        new_dataset = TankBindDataSet(f"{pre}/apr22_pdbbind_gvp_pocket_radius20", add_noise_to_com=add_noise_to_com)
        # load GIN embedding for compounds.
        new_dataset.compound_dict = torch.load(f"{pre}/pdbbind_compound_dict_with_LAS_distance_constraint_mask.pt")
        new_dataset.data = new_dataset.data.query("c_length < 100 and native_num_contact > 5").reset_index(drop=True)
        d = new_dataset.data
        only_native_train_index = d.query("use_compound_com and group =='train'").index.values
        train = new_dataset[only_native_train_index]
        # train = train
        train_index = d.query("group =='train'").index.values
        train_after_warm_up = new_dataset[train_index]

        # train = torch.utils.data.ConcatDataset([train1, train2])
        valid_index = d.query("use_compound_com and group =='valid'").index.values
        valid = new_dataset[valid_index]
        test_index = d.query("use_compound_com and group =='test'").index.values
        test = new_dataset[test_index]

        all_pocket_test_fileName = f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20/"
        all_pocket_test = TankBindDataSet(all_pocket_test_fileName)
        all_pocket_test.compound_dict = torch.load(f"{pre}/pdbbind_test_compound_dict_based_on_rdkit.pt")
        # info is used to evaluate the test set.
        info = None
        # info = pd.read_csv(f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)
    return train, train_after_warm_up, valid, test, all_pocket_test, info

# Added by Enzo
from torch_geometric.loader.dataloader import Collater
import torch
from torch_geometric.utils import to_dense_batch

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
        #remove collate_fn from kwargs
        if "collate_fn" in kwargs:
            kwargs.pop("collate_fn")
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
