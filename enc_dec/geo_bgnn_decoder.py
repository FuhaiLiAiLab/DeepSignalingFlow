import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch.nn import init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros

class BGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, node_num, num_edge, num_gene_edge, device):
        super(BGNNConv, self).__init__(aggr='add')
        self.node_num = node_num
        self.num_edge = num_edge
        self.num_gene_edge = num_gene_edge
        self.num_drug_edge = num_edge - num_gene_edge

        self.up_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.down_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias_proj = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        ### [edge_index, x] ###
        up_edge_index = edge_index
        up_x = self.up_proj(x)
        down_edge_index = torch.flipud(edge_index)
        down_x = self.down_proj(x)
        bias_x = self.bias_proj(x)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        # [up]
        up_row, up_col = up_edge_index
        up_deg = degree(up_col, x.size(0), dtype=x.dtype)
        up_deg_inv_sqrt = up_deg.pow(-1)
        up_norm = up_deg_inv_sqrt[up_col]
        # [down]
        down_row, down_col = down_edge_index
        down_deg = degree(down_col, x.size(0), dtype=x.dtype)
        down_deg_inv_sqrt = down_deg.pow(-1)
        down_norm = down_deg_inv_sqrt[down_col]
        # Check [ norm[0:59241] == norm[59455:118696] ]

        # Step 4-5: Start propagating messages.
        x_up = self.propagate(up_edge_index, x=up_x, norm=up_norm)
        x_down = self.propagate(down_edge_index, x=down_x, norm=down_norm)
        x_bias = bias_x
        # import pdb; pdb.set_trace()
        concat_x = torch.cat((x_up, x_down, x_bias), dim=-1)
        concat_x = F.normalize(concat_x, p=2, dim=-1)
        return concat_x

    def message(self, x_j, norm):
        # [x_j] has shape [E, out_channels]
        # import pdb; pdb.set_trace()
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] has shape [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out


class BGNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_edge, num_gene_edge, device):
        super(BGNNDecoder, self).__init__()
        self.node_num = node_num
        self.embedding_dim = embedding_dim
        self.device = device
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)

        self.x_norm = nn.BatchNorm1d(input_dim)

        self.parameter1 = torch.nn.Parameter(torch.randn(int(embedding_dim*3), decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge):
        # conv_first [input_dim, hidden_dim]
        conv_first = BGNNConv(in_channels=input_dim, out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # conv_block [hidden_dim, hidden_dim/3]
        conv_block = BGNNConv(in_channels=int(hidden_dim*3), out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # conv_last [hidden_dim, embedding_dim]
        conv_last = BGNNConv(in_channels=int(hidden_dim*3), out_channels=embedding_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index, drug_index):
        x_norm = self.x_norm(x)
        # import pdb; pdb.set_trace()
        x = self.conv_first(x_norm, edge_index)
        x = self.act2(x)

        x = self.conv_block(x, edge_index)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.act2(x)
        # import pdb; pdb.set_trace()
        # x = torch.reshape(x, (-1, self.node_num, self.embedding_dim))
        drug_index = torch.reshape(drug_index, (-1, 2))

        # EMBEDDING DECODER TO [ypred]
        batch_size, drug_num = drug_index.shape
        ypred = torch.zeros(batch_size, 1).to(device='cuda')
        for i in range(batch_size):
            drug_a_idx = int(drug_index[i, 0]) - 1
            drug_b_idx = int(drug_index[i, 1]) - 1
            drug_a_embedding = x[drug_a_idx]
            drug_b_embedding = x[drug_b_idx]
            product1 = torch.matmul(drug_a_embedding, self.parameter1)
            product2 = torch.matmul(product1, self.parameter2)
            product3 = torch.matmul(product2, torch.transpose(self.parameter1, 0, 1))
            output = torch.matmul(product3, drug_b_embedding.reshape(-1, 1))
            ypred[i] = output
        print(self.parameter1)
        print(torch.sum(self.parameter1))
        # print(self.parameter2)
        return ypred

    def loss(self, pred, label):
        pred = pred.to(device='cuda')
        label = label.to(device='cuda')
        loss = F.mse_loss(pred.squeeze(), label)
        # print(pred)
        # print(label)
        return loss