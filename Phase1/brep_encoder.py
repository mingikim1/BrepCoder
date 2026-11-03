import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch.glob import MaxPooling


class LayerNorm1d(nn.Module):
    """LayerNorm for Conv1d output: [batch, channels, length] -> normalize over channels"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        # x: [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, length, channels]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [batch, channels, length]
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm for Conv2d output: [batch, channels, H, W] -> normalize over channels"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        # x: [batch, channels, H, W]
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, channels]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [batch, channels, H, W]
        return x


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """Conv1d with LayerNorm and LeakyReLU"""
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        ),
        LayerNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """Conv2d with LayerNorm and LeakyReLU"""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        LayerNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.LayerNorm(out_features),  # LayerNorm for last dimension
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.layer_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.linears[i](h)
                h = self.layer_norms[i](h)
                h = F.relu(h)
            return self.linears[-1](h)


class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=128):
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        # x: [num_edges, L, channels] or [num_edges, channels, L]
        # Convert to [num_edges, channels, L] format for Conv1d
        if x.dim() == 3:
            if x.size(1) == self.in_channels:
                # Already in [num_edges, channels, L] format
                pass
            elif x.size(2) == self.in_channels:
                # [num_edges, L, channels] -> [num_edges, channels, L]
                x = x.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected input shape: {x.size()}, expected channels={self.in_channels}")
        
        num_edges = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(num_edges, -1)
        x = self.fc(x)
        return x


class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        output_dims=128,
    ):
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        # x: [num_nodes, H, W, channels] or [num_nodes, channels, H, W]
        # Convert to [num_nodes, channels, H, W] format for Conv2d
        if x.dim() == 4:
            if x.size(1) == self.in_channels:
                # Already in [num_nodes, channels, H, W] format
                pass
            elif x.size(3) == self.in_channels:
                # [num_nodes, H, W, channels] -> [num_nodes, channels, H, W]
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected input shape: {x.size()}, expected channels={self.in_channels}")
        
        num_nodes = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(num_nodes, -1)
        x = self.fc(x)
        return x



class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        super(_EdgeConv, self).__init__()
        self.proj = nn.Linear(node_feats, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.layer_norm = nn.LayerNorm(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        # Ensure inputs have matching dtype (critical for bf16 training)
        dtype = next(self.parameters()).dtype
        nfeat = nfeat.to(dtype=dtype)
        efeat = efeat.to(dtype=dtype)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(nfeat).any() or torch.isinf(nfeat).any():
            print(f"ERROR in _EdgeConv: NaN/Inf in nfeat! Shape: {nfeat.shape}")
        if torch.isnan(efeat).any() or torch.isinf(efeat).any():
            print(f"ERROR in _EdgeConv: NaN/Inf in efeat! Shape: {efeat.shape}")
        
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        
        # Check after projection
        if torch.isnan(proj1).any() or torch.isinf(proj1).any():
            print(f"ERROR in _EdgeConv: NaN/Inf in proj1 after self.proj!")
        if torch.isnan(proj2).any() or torch.isinf(proj2).any():
            print(f"ERROR in _EdgeConv: NaN/Inf in proj2 after self.proj!")
        
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        
        # Check after MLP
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"ERROR in _EdgeConv: NaN/Inf in h after MLP!")
        
        h = self.layer_norm(h)
        h = F.leaky_relu(h)
        return h


class _NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(
            in_feats=node_feats,
            out_feats=out_feats,
            edge_func=nn.Linear(edge_feats, node_feats * out_feats),
            aggregator_type="sum",
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(out_feats)
        self.mlp = _MLP(num_mlp_layers, out_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        # Ensure inputs have matching dtype (critical for bf16 training)
        dtype = next(self.parameters()).dtype
        nfeat = nfeat.to(dtype=dtype)
        efeat = efeat.to(dtype=dtype)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(nfeat).any() or torch.isinf(nfeat).any():
            print(f"ERROR in _NodeConv: NaN/Inf in nfeat! Shape: {nfeat.shape}")
        if torch.isnan(efeat).any() or torch.isinf(efeat).any():
            print(f"ERROR in _NodeConv: NaN/Inf in efeat! Shape: {efeat.shape}")
        
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        
        # Check after gconv
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"ERROR in _NodeConv: NaN/Inf in h after gconv!")
        
        h = self.mlp(h)
        
        # Check after MLP
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"ERROR in _NodeConv: NaN/Inf in h after MLP!")
        
        h = self.layer_norm(h)
        h = F.leaky_relu(h)
        return h


class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=128,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            # EdgeConv receives updated node features from NodeConv, so node_feats should be hidden_dim
            self.edge_conv_layers.append(
                _EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,
                    node_feats=hidden_dim,  # Use hidden_dim since it receives output from NodeConv
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

    def forward(self, g, h, efeat):
        # Check inputs
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"ERROR in UVNetGraphEncoder.forward: NaN/Inf in input h! Shape: {h.shape}")
        if torch.isnan(efeat).any() or torch.isinf(efeat).any():
            print(f"ERROR in UVNetGraphEncoder.forward: NaN/Inf in input efeat! Shape: {efeat.shape}")
        
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)
            # Check after node conv
            if torch.isnan(h).any() or torch.isinf(h).any():
                print(f"ERROR in UVNetGraphEncoder: NaN/Inf in h after node_conv_layers[{i}]!")
            
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)
            # Check after edge conv
            if torch.isnan(he).any() or torch.isinf(he).any():
                print(f"ERROR in UVNetGraphEncoder: NaN/Inf in he after edge_conv_layers[{i}]!")
            
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer = score_over_layer + self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer


class BrepEncoderWrapper(nn.Module):
    """
    Wrapper for B-Rep encoder that outputs [batch, max_nodes, dimension]
    Dynamically pads to the maximum number of nodes in the batch.
    
    Output dimension: node_emb_dim + graph_emb_dim (local + global features)
    - node_emb: Per-node features from message passing [num_nodes, hidden_dim]
    - graph_emb: Global graph features from MaxPooling [1, output_dim]
    - Each node gets: concat(node_emb, graph_emb) for local-global context
    
    This follows the original UV-Net Segmentation architecture where each node
    receives both its local embedding and the global graph embedding.
    """
    def __init__(
        self,
        curve_channels=6,
        surface_channels=7,
        curve_emb_dim=128,
        surface_emb_dim=128,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        num_mlp_layers=2,
        max_nodes_limit=2048,  # Upper limit for safety
        use_gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        concat_graph_emb=True,  # Whether to concat graph_emb to each node (UV-Net style)
    ):
        super(BrepEncoderWrapper, self).__init__()
        self.max_nodes_limit = max_nodes_limit
        self.hidden_dim = hidden_dim  # node_emb dimension
        self.graph_emb_dim = output_dim  # graph_emb dimension
        self.curve_emb_dim = curve_emb_dim
        self.surface_emb_dim = surface_emb_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.concat_graph_emb = concat_graph_emb
        
        # Final output dimension: node_emb + graph_emb (if concat) or just node_emb
        if concat_graph_emb:
            self.output_dim = hidden_dim + output_dim  # e.g., 128 + 128 = 256
        else:
            self.output_dim = hidden_dim  # e.g., 128
        
        # Curve and Surface encoders
        self.curve_encoder = UVNetCurveEncoder(
            in_channels=curve_channels,
            output_dims=curve_emb_dim
        )
        self.surface_encoder = UVNetSurfaceEncoder(
            in_channels=surface_channels,
            output_dims=surface_emb_dim
        )
        
        # Graph encoder - output_dim must equal hidden_dim for node features
        assert hidden_dim == output_dim, "hidden_dim must equal output_dim for node features"
        self.graph_encoder = UVNetGraphEncoder(
            input_dim=surface_emb_dim,
            input_edge_dim=curve_emb_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mlp_layers=num_mlp_layers,
        )
    
    def forward(self, batched_graphs):
        """
        Args:
            batched_graphs: List of DGL graphs
                Each graph should have:
                - ndata['x']: [num_nodes, surface_channels, H, W] - surface features
                - edata['x']: [num_edges, curve_channels, L] - curve features
        
        Returns:
            brep_feats: [batch_size, max_nodes, output_dim]
                        output_dim = hidden_dim + graph_emb_dim (if concat_graph_emb=True)
                        output_dim = hidden_dim (if concat_graph_emb=False)
            brep_padding_mask: [batch_size, max_nodes] (1 for valid, 0 for padding)
        """
        import dgl
        
        # Convert to list if needed
        if not isinstance(batched_graphs, list):
            graphs = dgl.unbatch(batched_graphs)
        else:
            graphs = batched_graphs
        
        batch_size = len(graphs)
        # Get device from model parameters (works with both regular and DDP models)
        device = next(self.parameters()).device
        # Get dtype from model parameters (for bf16/fp16 support)
        dtype = next(self.parameters()).dtype
        
        # Record number of nodes per graph for later unbatching
        num_nodes_list = [g.num_nodes() for g in graphs]
        num_edges_list = [g.num_edges() for g in graphs]
        
        # Batch all graphs into one super-graph for parallel processing
        # This is much more efficient than processing graphs one by one
        bg = dgl.batch(graphs).to(device)
        
        # Convert graph data to model dtype (bf16/fp16/fp32)
        if bg.num_edges() > 0 and 'x' in bg.edata:
            edge_feats_raw = bg.edata['x']
            # Check for NaN/Inf in raw edge features
            if torch.isnan(edge_feats_raw).any() or torch.isinf(edge_feats_raw).any():
                print(f"ERROR: NaN/Inf in raw edge_feats! Shape: {edge_feats_raw.shape}, NaN count: {torch.isnan(edge_feats_raw).sum()}, Inf count: {torch.isinf(edge_feats_raw).sum()}")
                import traceback
                traceback.print_stack()
            bg.edata['x'] = edge_feats_raw.to(dtype=dtype)
        if bg.num_nodes() > 0 and 'x' in bg.ndata:
            node_feats_raw = bg.ndata['x']
            # Check for NaN/Inf in raw node features
            if torch.isnan(node_feats_raw).any() or torch.isinf(node_feats_raw).any():
                print(f"ERROR: NaN/Inf in raw node_feats! Shape: {node_feats_raw.shape}, NaN count: {torch.isnan(node_feats_raw).sum()}, Inf count: {torch.isinf(node_feats_raw).sum()}")
                import traceback
                traceback.print_stack()
            bg.ndata['x'] = node_feats_raw.to(dtype=dtype)
        
        # Encode all edges (curves) at once - PARALLEL
        if bg.num_edges() > 0:
            edge_feats = bg.edata['x']
            # Check for NaN/Inf before encoding
            if torch.isnan(edge_feats).any() or torch.isinf(edge_feats).any():
                print("Warning: NaN/Inf in edge_feats before curve_encoder, replacing with zeros")
                edge_feats = torch.where(torch.isnan(edge_feats) | torch.isinf(edge_feats),
                                        torch.zeros_like(edge_feats), edge_feats)
            if self.use_gradient_checkpointing and self.training:
                efeat = checkpoint(self.curve_encoder, edge_feats, use_reentrant=False)
            else:
                efeat = self.curve_encoder(edge_feats)
            # Check for NaN/Inf after encoding
            if torch.isnan(efeat).any() or torch.isinf(efeat).any():
                print("Warning: NaN/Inf in efeat after curve_encoder, replacing with zeros")
                efeat = torch.where(torch.isnan(efeat) | torch.isinf(efeat),
                                  torch.zeros_like(efeat), efeat)
            efeat = efeat.to(dtype=dtype)
        else:
            efeat = torch.zeros(0, self.curve_emb_dim, device=device, dtype=dtype)
        
        # Encode all nodes (surfaces) at once - PARALLEL
        if bg.num_nodes() > 0:
            node_feats = bg.ndata['x']
            # Check for NaN/Inf before encoding
            if torch.isnan(node_feats).any() or torch.isinf(node_feats).any():
                print("Warning: NaN/Inf in node_feats before surface_encoder, replacing with zeros")
                node_feats = torch.where(torch.isnan(node_feats) | torch.isinf(node_feats),
                                       torch.zeros_like(node_feats), node_feats)
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(self.surface_encoder, node_feats, use_reentrant=False)
            else:
                h = self.surface_encoder(node_feats)
            # Check for NaN/Inf after encoding
            if torch.isnan(h).any() or torch.isinf(h).any():
                print("Warning: NaN/Inf in h after surface_encoder, replacing with zeros")
                h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
            h = h.to(dtype=dtype)
            
            # Graph encoder on batched graph - processes all graphs together
            # node_embs: [total_nodes, hidden_dim]
            # graph_emb: [batch_size, graph_emb_dim]
            node_embs, graph_emb = self.graph_encoder(bg, h, efeat)
            # Check for NaN/Inf after graph encoder
            if torch.isnan(node_embs).any() or torch.isinf(node_embs).any():
                print("Warning: NaN/Inf in node_embs after graph_encoder, replacing with zeros")
                node_embs = torch.where(torch.isnan(node_embs) | torch.isinf(node_embs),
                                      torch.zeros_like(node_embs), node_embs)
            if torch.isnan(graph_emb).any() or torch.isinf(graph_emb).any():
                print("Warning: NaN/Inf in graph_emb after graph_encoder, replacing with zeros")
                graph_emb = torch.where(torch.isnan(graph_emb) | torch.isinf(graph_emb),
                                      torch.zeros_like(graph_emb), graph_emb)
            node_embs = node_embs.to(dtype=dtype)
            graph_emb = graph_emb.to(dtype=dtype)
        else:
            node_embs = torch.zeros(0, self.hidden_dim, device=device, dtype=dtype)
            graph_emb = torch.zeros(batch_size, self.graph_emb_dim, device=device, dtype=dtype)
        
        # Unbatch node embeddings back to individual graphs
        all_node_feats = []
        node_offset = 0
        
        for i, num_nodes in enumerate(num_nodes_list):
            if num_nodes > 0:
                # Get this graph's node embeddings
                graph_node_embs = node_embs[node_offset:node_offset + num_nodes]
                
                if self.concat_graph_emb:
                    # Concat graph embedding to each node
                    # graph_emb[i]: [graph_emb_dim] -> [num_nodes, graph_emb_dim]
                    graph_emb_repeated = graph_emb[i:i+1].expand(num_nodes, -1)
                    graph_node_embs = torch.cat([graph_node_embs, graph_emb_repeated], dim=1)
                
                all_node_feats.append(graph_node_embs)
                node_offset += num_nodes
            else:
                all_node_feats.append(torch.zeros(0, self.output_dim, device=device, dtype=dtype))
        
        # Dynamically pad to max nodes in batch
        max_nodes = min(max(num_nodes_list) if num_nodes_list else 0, self.max_nodes_limit)
        
        # Build padding mask
        brep_padding_mask = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=device)
        
        # Create padded features by stacking
        padded_feats_list = []
        for i, node_feats in enumerate(all_node_feats):
            num_nodes = min(node_feats.size(0), max_nodes)
            if num_nodes > 0:
                if num_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - num_nodes, self.output_dim, device=device, dtype=dtype)
                    padded_feat = torch.cat([node_feats[:num_nodes], padding], dim=0)
                else:
                    padded_feat = node_feats[:num_nodes]
                brep_padding_mask[i, :num_nodes] = 1
            else:
                padded_feat = torch.zeros(max_nodes, self.output_dim, device=device, dtype=dtype)
            padded_feats_list.append(padded_feat)
        
        brep_feats = torch.stack(padded_feats_list, dim=0)
        
        return brep_feats, brep_padding_mask
