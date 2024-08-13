import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from torch_geometric.data import HeteroData
import torch_geometric
import glob
import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm


def compute_RMSD_batch(a, b, ptr):
    # correct rmsd calculation.

    distances=((a-b)**2).sum(dim=-1) # (compound_nodes_batch, 3) -> (compound_nodes_batch)
    sum_distances = torch_geometric.utils.segment(distances, ptr, reduce="mean")
    return torch.sqrt(sum_distances)



def compute_RMSD(a, b):
    # correct rmsd calculation.
    return np.sqrt((((a-b)**2).sum(axis=-1)).mean())
def compute_RMSD_batch(a, b, ptr):
    # correct rmsd calculation.

    distances=((a-b)**2).sum(dim=-1) # (compound_nodes_batch, 3) -> (compound_nodes_batch)
    sum_distances = torch_geometric.utils.segment(distances, ptr, reduce="mean")
    return torch.sqrt(sum_distances)

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    try:
        rid = AllChem.EmbedMolecule(mol,)
    except:
        print("weird error")
        rid = -1
    if rid == -1:
        try:
            rid = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        except:
            print("weird error")
            rid = -1
    if rid == -1:
        mol.Compute2DCoords()
        coords = mol.GetConformer().GetPositions()
        return mol, coords
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=0)
    coords = mol.GetConformer().GetPositions()
    mol = Chem.RemoveHs(mol)
    return mol, coords
def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()
def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=0, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit, coords = generate_conformation(mol_from_rdkit)
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)

from tqdm.notebook import tqdm
def distance_loss_function(epoch, x, batch):
    protein_nodes_xyz_batched, protein_nodes_xyz_mask = torch_geometric.utils.to_dense_batch(batch.protein_nodes_xyz, batch.protein_nodes_xyz_batch)
    x_batched, x_mask = torch_geometric.utils.to_dense_batch(x, batch.coords_batch)
    protein_compound_mask = torch.einsum("ij,ik->ijk", protein_nodes_xyz_mask, x_mask)
    dis = torch.cdist(protein_nodes_xyz_batched, x_batched)
    dis_clamp = torch.clamp(dis, max=10)
    dis_flat = dis_clamp[protein_compound_mask]
    interaction_loss = torch_geometric.utils.segment((dis_flat - batch.y_pred).abs(), batch.y_pred_ptr, reduce="mean")
    xx_mask = torch.einsum("ij,ik->ijk", x_mask, x_mask)
    config_dis = torch.cdist(x_batched, x_batched)[xx_mask]

    configuration_loss = 1 * (((config_dis-batch.compound_pair_dis_constraint).abs()))
    configuration_loss += 2 * ((1.22 - config_dis).relu())
    configuration_loss = torch_geometric.utils.segment(configuration_loss, batch.LAS_distance_constraint_mask_ptr, reduce="mean")
    if epoch < 500:
        loss = interaction_loss.sum()
    else:
        loss = 1 * (interaction_loss.sum() + 5e-3 * (epoch - 500) * configuration_loss.sum())
    # added by Enzo
    interaction_loss_sum = interaction_loss.sum()
    configuration_loss_sum = configuration_loss.sum() 
    # modification by Enzo: achieves 20 percent
    #loss = 1 * (interaction_loss_sum + 5e-2 * configuration_loss_sum)
    return loss, (interaction_loss_sum.detach(), configuration_loss_sum.detach())



def distance_optimize_compound_coords(batch, total_epoch=5000, loss_function=distance_loss_function, LAS_distance_constraint_mask=None, mode=0, show_progress=False):
    # random initialization. center at the protein center.
    # coords: shape n_compound_nodes, 3
    # y_pred: shape n_protein_nodes, n_compound_nodes
    # protein_nodes_xyz: shape n_protein_nodes, 3
    # compound_pair_dis_constraint: shape n_compound_nodes, n_compound_nodes
    # LAS_distance_constraint_mask: boolean tensor shape n_compound_nodes, n_compound_nodes
    batch = batch.to("cuda:0")

    # TODO: c_pred est le centre de la protéine. On obtient la valeur avec torch_scatter
    c_pred = torch_geometric.utils.segment(batch.protein_nodes_xyz, batch.protein_nodes_xyz_ptr, reduce="mean")
    c_pred = c_pred[batch.coords_batch]
    x = (5 * (2 * torch.randn_like(batch.coords) - 1) + c_pred.reshape(-1, 3)).detach().clone().requires_grad_(True)
    # modification by Enzo: achieves 20 percent
    optimizer = torch.optim.Adam([x], lr=0.01)
    # optimizer = torch.optim.Adam([x], lr=0.01)
    # optimizer = torch.optim.SGD([x], lr=1, momentum=0.9)
    loss_list = []
    rmsd_list = []
    progress_bar = tqdm(range(total_epoch))
    for epoch in progress_bar:
        optimizer.zero_grad()
        loss, (interaction_loss, configuration_loss) = loss_function(epoch, x, batch)
        #print(f"loss: {loss.item()} interaction loss: {interaction_loss.item()} configuration loss: {configuration_loss.item()}")
        loss.backward()
        
        optimizer.step()
        
        # Append the loss to the list
        loss_list.append(loss.item())
        
        # Compute RMSD
        rmsd = compute_RMSD_batch(batch.coords, x.detach(), batch.coords_ptr)
        rmsd_list += rmsd.detach().cpu().tolist()
        
        # Update the progress bar with the loss
        progress_bar.set_description(f"{interaction_loss.item():.2f} {configuration_loss.item():.2f} {loss.item():.2f} {rmsd.mean().item():.2f}")
        
    return x, loss_list, rmsd_list

def get_info_pred_distance(batch, n_repeat=1, total_epoch=5000, mode=0, show_progress=False):
    info = []
    if show_progress:
        it = tqdm(range(n_repeat))
    else:
        it = range(n_repeat)
    for repeat in it:
        # random initialization.
        # x = torch.rand(coords.shape, requires_grad=True)
        x, loss_list, rmsd_list = distance_optimize_compound_coords(batch, mode=mode, total_epoch=total_epoch, show_progress=False)
        rmsd = rmsd_list[-1]
        for i in range(len(batch.coords_ptr)-1):
            try:
                info.append([repeat, rmsd_list[batch.coords_ptr[i]:batch.coords_ptr[i+1]], float(loss_list[-1]), x[batch.coords_ptr[i]:batch.coords_ptr[i+1]].detach().cpu().numpy()])
            except:
                info.append([repeat, rmsd_list[batch.coords_ptr[i]:batch.coords_ptr[i+1]], 0, x[batch.coords_ptr[i]:batch.coords_ptr[i+1]].detach().cpu().numpy()])
    info = pd.DataFrame(info, columns=['repeat', 'rmsd', 'loss', 'coords'])
    return info

def get_info_pred_distance(batch, n_repeat=1, mode=0, show_progress=False):
    info = []
    if show_progress:
        it = tqdm(range(n_repeat))
    else:
        it = range(n_repeat)
    for repeat in it:
        # random initialization.
        # x = torch.rand(coords.shape, requires_grad=True)
        x, loss_list, rmsd_list = distance_optimize_compound_coords(batch, mode=mode, show_progress=False)
        rmsd = rmsd_list[-1]
        for i in range(len(batch.coords_ptr)-1):
            try:
                info.append([repeat, rmsd_list[batch.coords_ptr[i]:batch.coords_ptr[i+1]], float(loss_list[-1]), x[batch.coords_ptr[i]:batch.coords_ptr[i+1]].detach().cpu().numpy()])
            except:
                info.append([repeat, rmsd_list[batch.coords_ptr[i]:batch.coords_ptr[i+1]], 0, x[batch.coords_ptr[i]:batch.coords_ptr[i+1]].detach().cpu().numpy()])
    info = pd.DataFrame(info, columns=['repeat', 'rmsd', 'loss', 'coords'])
    return info

def below_threshold(x, threshold=5):
    return 100 * (x < threshold).sum() / len(x)
def simple_custom_description(data):
    t1 = data
    t2 = t1.describe()
    t3 = t1.iloc[:,1:].apply(below_threshold, threshold=2, axis=0).reset_index(name='2A').set_index('index').T
    t31 = t1.iloc[:,1:].apply(below_threshold, threshold=5, axis=0).reset_index(name='5A').set_index('index').T
    t32 = t1.iloc[:,1:].median().reset_index(name='median').set_index('index').T
    t4 = pd.concat([t2, t3, t31, t32]).loc[['mean', '25%', '50%', '75%', '5A', '2A', 'median']]
    t5 = t4.T.reset_index()
    return t5


import contextlib
from typing import Any, Callable, Generator, Iterable, List
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

def distribute_function(
    func: Callable,
    X: Iterable,
    n_jobs: int,
    description: str = "",
    total: int = 1,
    use_enumerate: bool = False,
    **kwargs,
) -> Any:
    """Distributes function `func` over iterable `X` using `n_jobs` cores.
    Args:
        func (Callable): function to be distributed
        X (Iterable): iterable over which the function is distributed
        n_jobs (int): number of cores to use
        description (str, optional): Description of the progress. Defaults to "".
        total (int, optional): Total number of elements in `X`. Defaults to 1.
    Returns:
        Any: result of the `func` applied to `X`.
    """
    if total == 1:
        total = len(X)  # type: ignore
    with tqdm_joblib(tqdm(desc=description, total=total)):
        if use_enumerate:
            Xt = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(func)(idx, x, **kwargs) for idx, x in enumerate(X)
            )
        else:
            Xt = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(func)(x, **kwargs) for x in X
            )
    return Xt


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Generator[tqdm, None, None]:
    """Context manager to patch joblib to report into tqdm progress bar.
    The code for the context manager is adapted from a Stack Overflow answer:
    https://stackoverflow.com/a/58936697
    Args:
        tqdm_object (tqdm): The tqdm object that will display the progress.
    Yields:
        tqdm_object (tqdm): The same tqdm object, after joblib has been patched.
    Example:
        with tqdm_joblib(tqdm(total=10)) as progress_bar:
            # joblib code here
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """Inner callback class for updating tqdm progress during joblib execution."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the callback, forwarding arguments to the parent class."""
            super().__init__(*args, **kwargs)
        def __call__(self, *args: Any, **kwargs: Any) -> None:
            """Update tqdm progress upon batch completion."""
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()