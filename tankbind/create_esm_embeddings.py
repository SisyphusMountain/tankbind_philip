import os
import sys
sys.path.append('/fs/pool/pool-marsot')
import pandas as pd
import torch
from esm import pretrained, FastaBatchedDataset
from Bio.PDB import PDBParser

import os
import pickle
from multiprocessing import Pool
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def print_cuda_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2:.2f} MB")
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

def get_sequences_from_pdbfile(file_path):
    if os.path.exists(f"{file_path}_sequence_result.pkl"):
        try:
            with open(f"{file_path}_sequence_result.pkl", "rb") as file:
                result =  pickle.load(file)
            return result
        except:
            None
    # Define the path for the result file

    # Parse the PDB file to extract sequences
    biopython_parser = PDBParser()
    try:
        structure = biopython_parser.get_structure("random_id", file_path)
        structure = structure[0]
        sequence = None
        for i, chain in enumerate(structure):
            seq = ""
            for residue in chain:
                if residue.get_resname() == "HOH":
                    continue
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == "CA":
                        c_alpha = list(atom.get_vector())
                    if atom.name == "N":
                        n = list(atom.get_vector())
                    if atom.name == "C":
                        c = list(atom.get_vector())
                if c_alpha is not None and n is not None and c is not None:  # only append residue if it is an amino acid
                    try:
                        seq += three_to_one[residue.get_resname()]
                    except Exception:
                        seq += "-"
                        print("encountered unknown AA: ", residue.get_resname(), " in the complex. Replacing it with a dash - .")
            
            if sequence is None:
                sequence = seq
            else:
                sequence += ":" + seq
        with open(f"{file_path}_sequence_result.pkl", "wb") as file:
            pickle.dump(sequence, file)
    except:
        raise Exception(f"Error parsing file {file_path}")
    return sequence

def create_ESM_embeddings(labels, sequences, model_dim="650m"):
    """
    Parameters
    ----------
    labels : list
        List of labels.
    sequences : list
        List of sequences.
    Returns
    -------
    lm_embedding : dict[str: torch.Tensor]
        List of ESM embeddings, indexed by label.
    """
    if model_dim == "650m":
        model_location = "esm2_t33_650M_UR50D"
        toks_per_batch = 2**15
    elif model_dim == "15B":
        model_location = "esm2_t48_15B_UR50D"
        toks_per_batch = 4096
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    repr_layers = [33]
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=0)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            print_cuda_memory_usage()
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1 : truncate_len + 1].clone()
    return embeddings

def get_all_ESM_embeddings(protein_paths, protein_names, model="650m", save_dir="/fs/pool/pool-marsot/tankbind_philip/TankBind/data/protein_remove_extra_chains_10A"):

    if not os.path.exists(f"{save_dir}/sequences_dict.pkl") or True:
        sequences_dict = {protein_names[i]: get_sequences_from_pdbfile(protein_paths[i]) for i in range(len(protein_paths))}
        labels_cleaned, sequences_cleaned = [], []
        for name, sequence in sequences_dict.items():
            s = sequence.split(':')
            sequences_cleaned.extend(s)
            labels_cleaned.extend([name + '_chain_' + str(j) for j in range(len(s))])
        with open(f"{save_dir}/sequences_dict.pkl", "wb") as file:
            pickle.dump(sequences_cleaned, file)
        with open(f"{save_dir}/labels_cleaned.pkl", "wb") as file:
            pickle.dump(labels_cleaned, file)
        with open(f"{save_dir}/protein_names.pkl", "wb") as file:
            pickle.dump(protein_names, file)
        with open(f"{save_dir}/sequences_dict.pkl", "wb") as file:
            pickle.dump(sequences_dict, file)
    else: 
        with open(f"{save_dir}/sequences_dict.pkl", "rb") as file:
            sequences_cleaned = pickle.load(file)
        with open(f"{save_dir}/labels_cleaned.pkl", "rb") as file:
            labels_cleaned = pickle.load(file)
        with open(f"{save_dir}/protein_names.pkl", "rb") as file:
            protein_names = pickle.load(file)
        with open(f"{save_dir}/sequences_dict.pkl", "rb") as file:
            sequences_dict = pickle.load(file)

    if os.path.exists(f"{save_dir}/esm_embeddings_{model}_intermediate.pkl"):
        with open(f"{save_dir}/esm_embeddings_{model}_intermediate.pkl", "rb") as file:
            results = pickle.load(file)
    else:
        results = create_ESM_embeddings(labels_cleaned, sequences_cleaned, model)
        with open(f"{save_dir}/esm_embeddings_{model}_intermediate.pkl", "wb") as file:
            pickle.dump(results, file)
    embeddings = {}
    for name in protein_names:
        try:
            embeddings[name] = torch.cat([results[name + '_chain_' + str(j)] for j in range(len(sequences_dict[name].split(':')))], dim=0)
        except:
            None
    return embeddings

def list_folders(directory):
    entries = os.listdir(directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return folders
import os

def list_files(directory):
    entries = os.listdir(directory)
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    return files

def get_pdbbind_paths_and_names(folder_path):
    df = pd.read_csv("/fs/pool/pool-marsot/tankbind_philip/TankBind/data/data.csv")
    protein_names = df["pdb"].tolist()
    protein_paths = [f"{folder_path}/{name}_protein.pdb" for name in protein_names]

    return protein_paths, protein_names

def populate_sequences_dict(folder_path):
    protein_names = list(set(list_folders(folder_path)))
    protein_paths = [f"{folder_path}/{protein}/{protein}_protein_chains_in_contact_with_ligand.pdb" for protein in protein_names]
    
    # Filter out the paths that don't exist
    protein_paths = [path for path in protein_paths if os.path.exists(path)]
    with Pool(processes=32) as pool:
        pool.map(get_sequences_from_pdbfile, protein_paths)
    return None

def get_pdbbind_ESM_embeddings(folder_path="/fs/pool/pool-marsot/tankbind_philip/TankBind/data/protein_remove_extra_chains_10A", model="650m"):
    import time
    start = time.time()
    protein_paths, protein_names = get_pdbbind_paths_and_names(folder_path)
    result = get_all_ESM_embeddings(protein_paths, protein_names, model=model)
    end = time.time()
    print(f"Time taken: {end - start}")

    with open(f"{folder_path}/esm_embeddings_{model}.pkl", "wb") as file:
        pickle.dump(result, file)

    return result

if __name__ == "__main__":
    get_pdbbind_ESM_embeddings()