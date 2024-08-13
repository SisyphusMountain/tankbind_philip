import numpy as np
import torch
from tqdm import tqdm
import os
import rdkit.Chem as Chem
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric
from bindbind.datasets.processing.ligand_features.tankbind_ligand_features import read_molecule, create_tankbind_ligand_features, get_LAS_distance_constraint_mask
from .helper import compute_RMSD, write_with_new_coords, generate_sdf_from_smiles_using_rdkit, get_info_pred_distance, simple_custom_description, distribute_function
import pandas as pd
from tankbind_philip_base.tankbind_philip.tankbind.data import TankBindDataSet
from tankbind_philip.TankBind.tankbind.data import TankBindDataLoader


def evaluate_model_val(model,
                       batch_size=8,
                       num_workers=8,
                       val_dataset_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/dataset/val_dataset",
                       full_dataset_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/dataset/dataset",
                       rdkit_folder="/fs/pool/pool-marsot/tankbind_philip/TankBind/data/rdkit_folder",
                       renumbered_ligands_folder="/fs/pool/pool-marsot/tankbind_philip/TankBind/data/renumber_atom_index_same_as_smiles",
                       recompute=False,
                       ):

    val_dataset_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/dataset/val_dataset"
    full_dataset_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/dataset/dataset"
    recompute=False
    rdkit_folder="/fs/pool/pool-marsot/tankbind_philip/TankBind/data/rdkit_folder"
    renumbered_ligands_folder="/fs/pool/pool-marsot/tankbind_philip/TankBind/data/renumber_atom_index_same_as_smiles"
    num_workers=8
    batch_size=8

    if not os.path.exists(f"{val_dataset_path}/processed"):
        dataset = TankBindDataSet(full_dataset_path)
        val_data = dataset.data[(dataset.data["group"]=="valid") & (~(val_dataset.data)["pdb"].str.endswith('_c'))]
        val_names = val_data["protein_name"].unique().tolist()
        val_compound_dict = {name:item for (name, item) in dataset.compound_dict.items() if name in val_names}
        val_protein_dict = {name:item for (name, item) in dataset.protein_dict.items() if name in val_names}
        val_dataset = TankBindDataSet(val_dataset_path, data=val_data,
                                        compound_dict=val_compound_dict,
                                        protein_dict=val_protein_dict)
    else:
        val_dataset = TankBindDataSet(val_dataset_path)
    device = model.device
    model.eval()
    val = val_dataset.data["protein_name"].unique().tolist()
    if recompute or not os.path.exists(f"{rdkit_folder}/compound_dict_based_on_rdkit.pt"):
        compound_dict = {}
        print("generating compound dictionary")
        for protein_name in tqdm(val):
            mol, _ = read_molecule(f"{renumbered_ligands_folder}/{protein_name}.sdf", None)
            smiles = Chem.MolToSmiles(mol)
            rdkit_mol_path = f"{rdkit_folder}/{protein_name}.sdf"
            generate_sdf_from_smiles_using_rdkit(smiles, rdkit_mol_path, shift_dis=0.0)
            mol, _ = read_molecule(rdkit_mol_path, None)
            compound_dict[protein_name] = create_tankbind_ligand_features(rdkit_mol_path, None, has_LAS_mask=True)
        torch.save(compound_dict, f"{rdkit_folder}/compound_dict_based_on_rdkit.pt")
    else:
        compound_dict = torch.load(f"{rdkit_folder}/compound_dict_based_on_rdkit.pt")

    data_loader = TankBindDataLoader(val_dataset,follow_batch=["protein_nodes_xyz", "coords", "y_pred", "y", "LAS_distance_constraint_mask", "compound_pair"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    affinity_pred_list = []
    y_pred_list = []
    print("predicting pairwise distances")
    for data in tqdm(data_loader):
        data = data.to(device)
        previous_index_start=0
        this_index_start=0
        protein_sizes = torch.diff(data["protein"].ptr)
        compound_sizes = torch.diff(data["compound"].ptr)

        with torch.no_grad():
            y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for i in range(data.batch_n):
            this_index_start += protein_sizes[i] * compound_sizes[i]
            y_pred_list.append((y_pred[previous_index_start:this_index_start]).detach().cpu())
            previous_index_start = this_index_start.clone()
    affinity_pred_list = torch.cat(affinity_pred_list)
    output_info_chosen = val_dataset.data
    output_info_chosen["affinity"] = affinity_pred_list
    output_info_chosen['dataset_index'] = range(len(output_info_chosen))

    chosen = output_info_chosen.loc[output_info_chosen.groupby(['protein_name'], sort=False)['affinity'].agg('idxmax')].reset_index()
    device = "cpu"
    compound_coordinates_dict = {}
    protein_coordinates_dict = {}
    for name in val:
        compound_coordinates_dict[name] = compound_dict[name]["tankbind_ligand_atom_coordinates"]
        
        protein_coordinates_dict[name] = val_dataset.protein_dict[name][0]
    max_compound_nodes = 0
    max_protein_nodes = 0
    list_mols = []
    list_complexes = []
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['protein_name']
        dataset_index = line['dataset_index']

        coords = val_dataset[dataset_index].coords
        protein_node_coordinates = val_dataset[dataset_index].node_xyz
        # if denormalize:
        #    protein_node_coordinates = denormalize_feature(protein_node_coordinates, "protein_node_coordinates")
        n_compound = coords.shape[0]
        n_protein = protein_node_coordinates.shape[0]
        y_pred = y_pred_list[dataset_index]
        y = val_dataset[dataset_index].dis_map
        rdkit_mol_path = f"{rdkit_folder}/{protein_name}.sdf"
        mol, _ = read_molecule(rdkit_mol_path, None)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool().flatten()
        max_compound_nodes = max(max_compound_nodes, n_compound)
        max_protein_nodes = max(max_protein_nodes, n_protein)
        cplx = HeteroData()
        cplx.protein_name = protein_name
        cplx.protein_nodes_xyz = protein_node_coordinates
        cplx.coords = coords
        cplx.y_pred = y_pred
        cplx.y = y
        cplx.LAS_distance_constraint_mask = LAS_distance_constraint_mask
        list_complexes.append(cplx)
        list_mols.append(mol)

    dataloader = DataLoader(list_complexes, batch_size=chosen.shape[0], shuffle=False,
                            follow_batch=["protein_nodes_xyz", "coords", "y_pred", "y", "LAS_distance_constraint_mask"])

    batch = next(iter(dataloader))
    coords_batched, coords_mask = torch_geometric.utils.to_dense_batch(batch.coords, batch.coords_batch)
    coords_pair_mask = torch.einsum("ij,ik->ijk", coords_mask, coords_mask)
    compound_pair_dis_constraint = torch.cdist(coords_batched, coords_batched)[coords_pair_mask]
    batch.compound_pair_dis_constraint = compound_pair_dis_constraint

    pred_dist_info = get_info_pred_distance(batch,
                                n_repeat=1, show_progress=False)

    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['protein_name']
        toFile = f'{rdkit_folder}/{protein_name}_tankbind_chosen.sdf'
        new_coords = pred_dist_info['coords'].iloc[idx].astype(np.double)
        write_with_new_coords(list_mols[idx], new_coords, toFile)

    ligand_metrics = []
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['protein_name']
        mol, _ = read_molecule(f"{renumbered_ligands_folder}/{protein_name}.sdf", None)
        mol_pred, _ = read_molecule(f"{rdkit_folder}/{protein_name}_tankbind_chosen.sdf", None) # tankbind_chosen is the compound with predicted coordinates assigned by write_with_new_coords

        sm = Chem.MolToSmiles(mol)
        mol_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
        mol = Chem.RenumberAtoms(mol, mol_order)
        mol = Chem.RemoveHs(mol)
        true_ligand_pos = np.array(mol.GetConformer().GetPositions())

        sm = Chem.MolToSmiles(mol_pred)
        mol_order = list(mol_pred.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
        mol_pred = Chem.RenumberAtoms(mol_pred, mol_order)
        mol_pred = Chem.RemoveHs(mol_pred)
        mol_pred_pos = np.array(mol_pred.GetConformer().GetPositions())

        rmsd = np.sqrt(((true_ligand_pos - mol_pred_pos) ** 2).sum(axis=1).mean(axis=0))
        com_dist = compute_RMSD(mol_pred_pos.mean(axis=0), true_ligand_pos.mean(axis=0))
        ligand_metrics.append([protein_name, rmsd, com_dist,])

    d = pd.DataFrame(ligand_metrics, columns=['pdb', 'TankBind_RMSD', 'TankBind_COM_DIST',])
    return simple_custom_description(d)