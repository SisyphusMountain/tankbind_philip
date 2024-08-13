import sys
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_dense_batch
from torch.nn import Linear

import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
import lightning as L
#from GATv2 import GAT
from GINv2 import GIN
import xformers.ops as xops
from lr_schedulers import get_lr_schedule_cls
import torchmetrics

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x



class GVP_embedding(nn.Module):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):

        super(GVP_embedding, self).__init__()
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        seq = self.W_s(seq)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out


def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    return pair_dis_one_hot

class TriangleProteinToCompound(torch.nn.Module):
    def __init__(self, embedding_channels=256, c=128, hasgate=True):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)
        self.hasgate = hasgate
        if hasgate:
            self.gate_linear = Linear(embedding_channels, c)
        self.linear = Linear(embedding_channels, c)
        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j, 1
        z = self.layernorm(z)
        if self.hasgate:
            ab = self.gate_linear(z).sigmoid() * self.linear(z) * z_mask
        else:
            ab = self.linear(z) * z_mask
        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab)
        block2 = torch.einsum("bikc,bjkc->bijc", ab, compound_pair)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

class TriangleProteinToCompound_v2(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels, bias=False)
        self.layernorm_c = torch.nn.LayerNorm(c, bias=False)

        # self.gate_linear1 = Linear(embedding_channels, c)
        # self.gate_linear2 = Linear(embedding_channels, c)
        # modification by Enzo to remove biases. (hypothesis: biases make the outputs dependent on padding)
        self.gate_linear1 = Linear(embedding_channels, c, bias=False)
        self.gate_linear2 = Linear(embedding_channels, c, bias=False)

        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        # print(g.shape, block1.shape, block2.shape)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

# class Self_Attention(nn.Module):
#     def __init__(self, hidden_size,num_attention_heads=8,drop_rate=0.5):
#         super().__init__()
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         self.dp = nn.Dropout(drop_rate)
#         self.ln = nn.LayerNorm(hidden_size)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self,q,k,v,attention_mask=None,attention_weight=None):
#         q = self.transpose_for_scores(q)
#         k = self.transpose_for_scores(k)
#         v = self.transpose_for_scores(v)
#         attention_scores = torch.matmul(q, k.transpose(-1, -2))

#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # attention_probs = self.dp(attention_probs)
#         if attention_weight is not None:
#             attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
#             # if self.training:
#             #     top_mask = (attention_weight_sorted_sorted<np.random.randint(28,45))
#             # else:
#             top_mask = (attention_weight_sorted_sorted<32)
#             attention_probs = attention_probs * top_mask
#             # attention_probs = attention_probs * attention_weight
#             attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
#         # print(attention_probs.shape,v.shape)
#         # attention_probs = self.dp(attention_probs)
#         outputs = torch.matmul(attention_probs, v)

#         outputs = outputs.permute(0, 2, 1, 3).contiguous()
#         new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
#         outputs = outputs.view(*new_output_shape)
#         outputs = self.ln(outputs)
#         return outputs


class FastTriangleSelfAttention(nn.Module):
    def __init__(self, embedding_channels, num_attention_heads):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels, bias=False)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = embedding_channels // num_attention_heads
        self.linear_qkv = nn.Linear(embedding_channels, 3*embedding_channels, bias=False)
        self.output_linear = nn.Linear(embedding_channels, embedding_channels)
        self.g = nn.Linear(embedding_channels, embedding_channels)
    def forward(self, z, z_mask_attention_float, z_mask):
        """
        Parameters
        ----------
        z: torch.Tensor of shape [batch, n_protein, n_compound, embedding_channels]
        z_mask: torch.Tensor of shape [batch*n_protein*num_attention_heads, n_compound, n_compound] saying which coefficients
            correspond to actual data. (we take this weird shape because scaled_dot_product_attention
            requires it). We take it to be float("-inf") where we want to mask.
        Returns
        -------
        """
        z = self.layernorm(z)
        batch_size, n_protein, n_compound, embedding_channels = z.shape
        z = z.reshape(batch_size*n_protein, n_compound, embedding_channels)
        q, k, v = self.linear_qkv(z).chunk(3, dim=-1)
        q = q.view(batch_size*n_protein, n_compound, self.num_attention_heads, self.attention_head_size).contiguous()
        k = k.view(batch_size*n_protein, n_compound, self.num_attention_heads, self.attention_head_size).contiguous()
        v = v.view(batch_size*n_protein, n_compound, self.num_attention_heads, self.attention_head_size).contiguous()
        attention_coefficients = xops.memory_efficient_attention(query=q,
                                                key=k,
                                                value=v,
                                                attn_bias=z_mask_attention_float.to("cuda:0")) # shape [batch*protein_nodes, compound_nodes, n_heads, embedding//n_heads]        

        attention_output = attention_coefficients.view(batch_size, n_protein, n_compound, embedding_channels)
        g = self.g(z).sigmoid()
        output = g * attention_output.view(batch_size*n_protein, n_compound, embedding_channels)

        output = self.output_linear(output.view(batch_size, n_protein, n_compound, embedding_channels))*z_mask.unsqueeze(-1).to('cuda:0')
        return output

class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        # self.layernorm_c = torch.nn.LayerNorm(c)

        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        # new_z = torch.zeros(z.shape, device=z.device)
        z_i = z
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        # q, k, v of shape b, j, h, c
        q = self.reshape_last_dim(self.linear_q(z_i)) #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        # attention_probs = self.dp(attention_probs)
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        # output of shape b, j, embedding.
        # z[:, i] = output
        z = output
        # print(g.shape, block1.shape, block2.shape)
        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        return z


class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, embedding_channels)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z



class TankBindModel(torch.nn.Module):
    def __init__(self,
                 hidden_channels=128,
                 embedding_channels=128,
                 c=128,
                 fast_attention=True,
                 mode=0,
                 protein_embed_mode=1, 
                 compound_embed_mode=1,
                 n_trigonometry_module_stack=5,
                 protein_bin_max=30,
                 readout_mode=2,
                 distogram_bins=None,
                 use_esm=False):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.protein_bin_max = protein_bin_max
        self.mode = mode
        self.protein_embed_mode = protein_embed_mode
        self.compound_embed_mode = compound_embed_mode
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.readout_mode = readout_mode
        self.n_heads=4
        self.use_esm = use_esm
        # Added by Enzo
        self.fast_attention = fast_attention
        if protein_embed_mode == 0:
            self.conv_protein = GNN(hidden_channels, embedding_channels)
            self.conv_compound = GNN(hidden_channels, embedding_channels)
            # self.conv_protein = SAGEConv((-1, -1), embedding_channels)
            # self.conv_compound = SAGEConv((-1, -1), embedding_channels)
        if protein_embed_mode == 1:
            if use_esm:
                conv_protein_input_dim = 1286
            else:
                conv_protein_input_dim = 6
            self.conv_protein = GVP_embedding((conv_protein_input_dim, 3), (embedding_channels, 16), 
                                              (32, 1), (32, 1), seq_in=True)
            

        if compound_embed_mode == 0:
            self.conv_compound = GNN(hidden_channels, embedding_channels)
        elif compound_embed_mode == 1:
            self.conv_compound = GIN(input_dim = 56, hidden_dims = [128,56,embedding_channels], edge_input_dim = 19, concat_hidden = False)

        if mode == 0:
            self.protein_pair_embedding = nn.Embedding(16, c)
            self.compound_pair_embedding = Linear(16, c)
            self.protein_to_compound_list = []
            self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
            if fast_attention:
                self.triangle_self_attention_list = nn.ModuleList([FastTriangleSelfAttention(embedding_channels=embedding_channels, num_attention_heads=4) for _ in range(n_trigonometry_module_stack)])
            else:
                self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels) for _ in range(n_trigonometry_module_stack)])
            self.tranistion = Transition(embedding_channels=embedding_channels, n=4)

        self.linear = Linear(embedding_channels, 1)
        self.linear_energy = Linear(embedding_channels, 1)
        if readout_mode == 2:
            self.gate_linear = Linear(embedding_channels, 1)
        # self.gate_linear = Linear(embedding_channels, 1)
        self.bias = torch.nn.Parameter(torch.ones(1))
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.25)
        self.distogram_bins = distogram_bins
        if self.distogram_bins is not None:
            self.distogram_head = Linear(embedding_channels, self.distogram_bins)
    def forward(self, data):
        # Added by Enzo
        max_dim_divisible_by_8_protein = data.max_dim_divisible_by_8_protein
        max_dim_divisible_by_8_compound = data.max_dim_divisible_by_8_compound
        if self.protein_embed_mode == 0:
            x = data['protein'].x.float()
            edge_index = data[("protein", "p2p", "protein")].edge_index
            protein_batch = data['protein'].batch
            protein_out = self.conv_protein(x, edge_index)
        if self.protein_embed_mode == 1:
            nodes = (data['protein']['node_s'], data['protein']['node_v'])
            edges = (data[("protein", "p2p", "protein")]["edge_s"], data[("protein", "p2p", "protein")]["edge_v"])
            protein_batch = data['protein'].batch
            protein_out = self.conv_protein(nodes, data[("protein", "p2p", "protein")]["edge_index"], edges, data.seq)

        if self.compound_embed_mode == 0:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index
            compound_batch = data['compound'].batch
            compound_out = self.conv_compound(compound_x, compound_edge_index)
        elif self.compound_embed_mode == 1:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index.T
            # compound_edge_index = data[("compound", "c2c", "compound")].edge_index
            compound_edge_feature = data[("compound", "c2c", "compound")].edge_attr
            edge_weight = data[("compound", "c2c", "compound")].edge_weight
            compound_batch = data['compound'].batch
            # Enzo : print dimensions
            #print(f"{compound_edge_index.shape=}, {edge_weight.shape=}, {compound_edge_feature.shape=}, {compound_x.shape=}")
            compound_out = self.conv_compound(compound_edge_index,edge_weight,compound_edge_feature,compound_x.shape[0],compound_x)['node_feature']
    
        # protein_batch version could further process b matrix. better than for loop.
        # protein_out_batched of shape b, n, c
        protein_out_batched, protein_out_mask = to_dense_batch(protein_out, protein_batch, max_num_nodes=max_dim_divisible_by_8_protein)
        compound_out_batched, compound_out_mask = to_dense_batch(compound_out, compound_batch, max_num_nodes=max_dim_divisible_by_8_compound)
        batch_n = data.batch_n
        z_mask = torch.einsum("bi,bj->bij", protein_out_mask, compound_out_mask)
        z_mask_attention = torch.einsum("bik, bq-> biqk", z_mask, compound_out_mask).reshape(batch_n*protein_out_batched.shape[1], max_dim_divisible_by_8_compound, max_dim_divisible_by_8_compound).unsqueeze(1).expand(-1, self.n_heads, -1, -1).contiguous()
        z_mask_attention = torch.where(z_mask_attention, 0.0, -10.0**6)
        z_mask_flat = torch.arange(
            start=0, end=z_mask.numel(), device=self.device
        ).view(z_mask.shape)[z_mask]
        protein_square_mask = torch.einsum("bi,bj->bij", protein_out_mask, protein_out_mask)

        node_xyz = data.node_xyz

        p_coords_batched, p_coords_mask = to_dense_batch(node_xyz, protein_batch)
        # c_coords_batched, c_coords_mask = to_dense_batch(coords, compound_batch)

        protein_pair = data["protein", "p2p", "protein"].pairwise_representation
        
        # compound_pair = get_pair_dis_one_hot(c_coords_batched, bin_size=1, bin_min=-0.5, bin_max=15)
        compound_pair_batched, compound_pair_batched_mask = data["compound", "p2p", "compound"].pairwise_representation, data["compound", "p2p", "compound"].pairwise_representation_mask

        batch_n = compound_pair_batched.shape[0]
        # max_compound_size_square = compound_pair_batched.shape[1]
        # max_compound_size = int(max_compound_size_square**0.5)
        # assert (max_compound_size**2 - max_compound_size_square)**2 < 1e-4
        # compound_pair = torch.zeros((batch_n, max_compound_size, max_compound_size, 16)).to(data.compound_pair.device)
        # for i in range(batch_n):
        #     one = compound_pair_batched[i]
        #     compound_size_square = (data.compound_pair_batch == i).sum()
        #     compound_size = int(compound_size_square**0.5)
        #     compound_pair[i,:compound_size, :compound_size] = one[:compound_size_square].reshape(
        #                                                         (compound_size, compound_size, -1))
        protein_pair = self.protein_pair_embedding(protein_pair)
        compound_pair = self.compound_pair_embedding(data["compound", "p2p", "compound"].pairwise_representation.float())        # b = torch.einsum("bik,bjk->bij", protein_out_batched, compound_out_batched).flatten()

        protein_out_batched = self.layernorm(protein_out_batched)
        compound_out_batched = self.layernorm(compound_out_batched)
        # z of shape, b, protein_length, compound_length, channels.
        z = torch.einsum("bik,bjk->bijk", protein_out_batched, compound_out_batched)
        # z_mask = torch.einsum("bi,bj->bij", protein_out_mask, compound_out_mask)

        # print(protein_pair.shape, compound_pair.shape, b.shape)
        if self.mode == 0:
            for _ in range(1):
                for i_module in range(self.n_trigonometry_module_stack):
                    z = z + self.dropout(self.protein_to_compound_list[i_module](z, protein_pair, compound_pair, z_mask.unsqueeze(-1)))
                    if self.fast_attention:
                        z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask_attention, z_mask))
                    else:
                        z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask))
                    z = self.tranistion(z)
        # batch_dim = z.shape[0]

        b = self.linear(z).squeeze(-1)

        y_pred = b.flatten()[z_mask_flat]
        y_pred = y_pred.sigmoid() * 10   # normalize to 0 to 10.
        if self.readout_mode == 0:
            pair_energy = self.linear_energy(z).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        if self.readout_mode == 1:
            # valid_interaction_z = (z * z_mask.unsqueeze(-1)).mean(axis=(1, 2))
            valid_interaction_z = (z * z_mask.unsqueeze(-1)).sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2)).unsqueeze(-1)
            affinity_pred = self.linear_energy(valid_interaction_z).squeeze(-1)
            # print("z shape", z.shape, "z_mask shape", z_mask.shape,   "valid_interaction_z shape", valid_interaction_z.shape, "affinity_pred shape", affinity_pred.shape)
        if self.readout_mode == 2:
            pair_energy = (self.gate_linear(z).sigmoid() * self.linear_energy(z)).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        if self.distogram_bins is not None:
            distogram_pred = self.distogram_head(z)
            distogram_pred[~z_mask] = -10.0**6
            distogram_pred = distogram_pred.flatten(end_dim=-2)[z_mask_flat]
            return y_pred, affinity_pred, distogram_pred
        return y_pred, affinity_pred

class TankBind(L.LightningModule):
    def __init__(self,
                 criterion=None,
                 affinity_criterion=None, 
                 constant_affinity_coeff=None,
                 decoy_gap=1.0,
                 lr=1e-5,
                 scheduler=None,
                 n_trigonometry_module_stack=5,
                 warmup_epochs=0,
                 max_epochs=100,
                 use_distogram_loss=False,
                 distogram_coefficient=1.0,
                 distogram_bins=20,
                 use_esm=False):
        super().__init__()
        if use_distogram_loss:
            self.model = TankBindModel(distogram_bins=distogram_bins,
                                       n_trigonometry_module_stack=n_trigonometry_module_stack,
                                       use_esm=use_esm)
        else:
            self.model = TankBindModel(n_trigonometry_module_stack=n_trigonometry_module_stack,
                                                           use_esm=use_esm)
        self.criterion = criterion
        self.affinity_criterion = affinity_criterion
        self.constant_affinity_coeff = constant_affinity_coeff
        self.decoy_gap = decoy_gap
        self.lr = lr
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.use_distogram_loss = use_distogram_loss
        self.distogram_criterion = distogram_loss
        self.distogram_coefficient = distogram_coefficient
        self.distogram_bins = distogram_bins
        # metrics.
        # We log the similarities for all distances, and for distances below 10. That's why we have 2 metrics for each metric.
        self.concordance_corr_coeff = nn.ModuleList([torchmetrics.ConcordanceCorrCoef() for _ in range(2)])
        self.cosine_sim = nn.ModuleList([torchmetrics.CosineSimilarity() for _ in range(2)])
        self.csi = torchmetrics.regression.CriticalSuccessIndex(9.9)
        self.explained_variance = nn.ModuleList([torchmetrics.ExplainedVariance() for _ in range(2)])
        self.kl_div = nn.ModuleList([self.KL_div(log_prob=True) for _ in range(2)])
        self.mae = nn.ModuleList([torchmetrics.MeanAbsoluteError() for _ in range(2)])
        self.mape = nn.ModuleList([torchmetrics.MeanAbsolutePercentageError() for _ in range(2)])
        self.mse = nn.ModuleList([torchmetrics.MeanSquaredError() for _ in range(2)])
        self.pearson = nn.ModuleList([torchmetrics.PearsonCorrCoef() for _ in range(2)])
        self.r2 = nn.ModuleList([torchmetrics.R2Score() for _ in range(2)])
        self.rse = nn.ModuleList([torchmetrics.RelativeSquaredError() for _ in range(2)])
        self.spearman = nn.ModuleList([torchmetrics.SpearmanCorrCoef() for _ in range(2)])
        self.entropy = nn.ModuleList([torchmetrics.Entropy() for _ in range(2)])
        self.kl_div = torchmetrics.EntropyMetric()
    def forward(self, batch):
        return self.model(batch)
    def training_step(self, batch, batch_idx):
        self.model.train()
        if not self.use_distogram_loss:
            y_pred, affinity_pred = self.model(batch)
        else:
            y_pred, affinity_pred, distogram_pred = self.model(batch)
        dis_map = batch.dis_map
        y_pred = y_pred[batch.equivalent_native_y_mask]
        dis_map = dis_map[batch.equivalent_native_y_mask]
        loss = self.criterion(y_pred, dis_map)
        affinity_loss = self.constant_affinity_coeff * self.affinity_criterion(affinity_pred,
                                                                               batch.affinity,
                                                                               batch.is_equivalent_native_pocket,
                                                                               decoy_gap=self.decoy_gap)
        total_loss = loss + affinity_loss
        dis_map_below_10 = dis_map < 10-1e-3
        if dis_map_below_10.sum() > 0:
            r2 = torchmetrics.functional.r2_score(y_pred[dis_map_below_10], dis_map[dis_map_below_10])
            self.log('train_r2', r2, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if self.use_distogram_loss:
            distogram_pred = distogram_pred[batch.equivalent_native_y_mask]
            dist_loss = self.distogram_criterion(distogram_pred, dis_map.clamp(0,10), n_bins=self.distogram_bins)
            if distogram_pred.numel() > 0:
                total_loss += dist_loss*self.distogram_coefficient
            else:
                dist_loss = torch.tensor(0.0, device=distogram_pred.device)
            self.log('train_distogram_loss', dist_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_mse', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) # mse is nan when input tensors are empty (none of the pockets contains the ligand so equivalent_native_y_mask is empty)
        self.log('train_affinity_loss', affinity_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return total_loss
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if not self.use_distogram_loss:
            y_pred, affinity_pred = self.model(batch)
        else:
            y_pred, affinity_pred, distogram_pred = self.model(batch)
        dis_map = batch.dis_map
        y_pred = y_pred[batch.equivalent_native_y_mask]
        dis_map = dis_map[batch.equivalent_native_y_mask]
        loss = self.criterion(y_pred, dis_map)
        affinity_loss = self.constant_affinity_coeff * self.affinity_criterion(affinity_pred,
                                                                               batch.affinity,
                                                                               batch.is_equivalent_native_pocket,
                                                                               decoy_gap=self.decoy_gap)
        total_loss = loss + affinity_loss
        dis_map_below_10 = dis_map < 10-1e-3
        r2 = torchmetrics.functional.r2_score(y_pred[dis_map_below_10], dis_map[dis_map_below_10])
        self.log('val_r2', r2, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.use_distogram_loss:
            dist_loss = self.distogram_criterion(distogram_pred, dis_map, n_bins=self.distogram_bins)
            total_loss += self.distogram_coefficient*dist_loss
            self.log('val_distogram_loss', dist_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_mse', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_affinity_loss', affinity_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.scheduler is not None:
            get_scheduler = get_lr_schedule_cls(self.scheduler)
            scheduler = get_scheduler(optimizer, self.warmup_epochs, self.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

def get_model(mode, logging, device):
    if mode == 0:
        logging.info("5 stack, readout2, pred dis map add self attention and GVP embed, compound model GIN")
        model = IaBNet_with_affinity().to(device)
    return model

def distogram_loss(y_pred, y_target, min_bin=0.0, max_bin=10.0001, n_bins=32):
    """Measure the accuracy of the binding site prediction.
    We use a distogram loss to penalize the prediction being too far from the target.
    Parameters
    ----------
    y_pred: torch.Tensor: shape [n_residues*n_atoms, n_bins]
        Predicted pairwise distances between atoms.
    y_target: torch.Tensor: shape [n_residues*n_atoms]
        Target pairwise distances between atoms.
    min_bin: float
        Minimum distance for the first bin.
    max_bin: float
        Maximum distance for the last bin.
    n_bins: int
        Number of bins in the distogram.

    Returns
    -------
    loss: torch.Tensor
        The computed distogram loss.
    """
    # Create bin edges
    bin_edges = torch.linspace(min_bin, max_bin, n_bins, device=y_pred.device)
    
    # Discretize the target distances
    y_target_binned = torch.bucketize(y_target, bin_edges).long()
    # import IPython; IPython.embed()
    # Compute CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(y_pred, y_target_binned)
    #raise Exception(f"{y_pred.shape}, {y_target_binned.max()}")

    return loss

from torchmetrics import Metric
from torch import Tensor
import torch.nn.functional as F

class EntropyMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_entropy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: Tensor) -> None:
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        self.total_entropy += torch.sum(entropy)
        self.total_samples += logits.size(0)

    def compute(self) -> Tensor:
        return self.total_entropy / self.total_samples