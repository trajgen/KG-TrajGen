import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import pandas as pd
import dgl
from dgl.nn.pytorch import GraphConv
import numpy as np
import torch as th
from torch import nn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl import apply_each
from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
from math import sqrt
class WeightedGATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    # >>> g = ... # a DGLGraph
    # >>> g = dgl.add_self_loop(g)
    #
    # Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    # since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    # to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    # A common practise to handle this is to filter out the nodes with zero-in-degree when use
    # after conv.
    #
    # Examples
    # --------
    # >>> import dgl
    # >>> import numpy as np
    # >>> import torch as th
    # >>> from dgl.nn import GATConv
    #
    # >>> # Case 1: Homogeneous graph
    # >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    # >>> g = dgl.add_self_loop(g)
    # >>> feat = th.ones(6, 10)
    # >>> gatconv = GATConv(10, 2, num_heads=3)
    # >>> res = gatconv(g, feat)
    # >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    # >>> # Case 2: Unidirectional bipartite graph
    # >>> u = [0, 1, 0, 0, 1]
    # >>> v = [0, 1, 2, 3, 2]
    # >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    # >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    # >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    # >>> gatconv = GATConv((5,10), 2, 3)
    # >>> res = gatconv(g, (u_feat, v_feat))
    # >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(WeightedGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.weight_linear = nn.Linear(1, 1)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # fuse the weights of edges
            # print(graph.edata['weight'].shape, graph.edata['weight'].T.shape)
            _w = self.weight_linear(graph.edata['weight'].unsqueeze(1))
            # print(e.shape, _w.shape)
            e = e * _w.unsqueeze(2) # (num_edges, num_heads, hidden_dim) * (num_edges, 1, 1)

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, device, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.device = device
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #


        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    But, it's hopefully much more readable! (and of similar performance)
    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, device, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, device, concat, activation,
                         dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:
        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted.to(device=self.device), exp_scores_per_edge.to(device=self.device))

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index.to(device=self.device))

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted.to(device=self.device), nodes_features_proj_lifted_weighted.to(device=self.device))

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index.to(device=self.device))
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index.to(device=self.device))
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index.to(device=self.device))

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


def to_hetero_feat(h, type, name):
    """Feature convert API.

    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.

    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.

    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph

    Example
    -------

    >>> h = torch.tensor([[1, 2, 3],
                          [1, 1, 1],
                          [0, 2, 1],
                          [1, 3, 3],
                          [2, 1, 1]])
    >>> print(h.shape)
    torch.Size([5, 3])
    >>> type = torch.tensor([0, 1, 0, 0, 1])
    >>> name = ['author', 'paper']
    >>> h_dict = to_hetero_feat(h, type, name)
    >>> print(h_dict)
    {'author': tensor([[1, 2, 3],
    [0, 2, 1],
    [1, 3, 3]]), 'paper': tensor([[1, 1, 1],
    [2, 1, 1]])}

    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[th.where(type == index)]

    return h_dict

#
class R_GAT(nn.Module):
    def __init__(self, etypes, in_size, hid_size, out_size, n_heads=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.HeteroGraphConv(
                {  #GATConv 的输出将是一个形状为 [num_nodes, num_heads, out_features]
                    etype: dglnn.GATConv(in_size, hid_size , n_heads, activation=F.elu)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size*n_heads, hid_size, 1, activation=F.elu)
                    for etype in etypes
                }
            )
        )
        # self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hid_size, out_size)  # Should be HeteroLinear

    def forward(self,g_base,x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(g_base, h)
            if l == len(self.layers) - 1:  # last layer
                h["Road"] = h["Road"].mean(1)
            else:  # other layer(s)
                h["Road"] = h["Road"].flatten(1)
        # 假设我们关注的是"Road"类型的节点
        return self.linear(h["Road"])


class SimpleHGN(nn.Module):
   def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                num_layers, heads, feat_drop=0.0, negative_slope=0.05,
                residual=True, beta=0.05):

       super(SimpleHGN, self).__init__()
       #边的嵌入维度，边的类型数量，输入特征维度，隐藏层维度，输出维度，层的数量，头数，dropout
       self.num_layers = num_layers
       self.hgn_layers = nn.ModuleList()
       self.activation = F.elu

       # input projection (no residual)
       self.hgn_layers.append(
           SimpleHGNConv(
               edge_dim,
               in_dim,
               hidden_dim,
               heads,
               num_etypes,
               feat_drop,
               negative_slope,
               False,
               self.activation,
               beta=beta,
           )
       )
       # hidden layers
       for l in range(1, num_layers - 1):  # noqa E741
           # due to multi-head, the in_dim = hidden_dim * num_heads
           self.hgn_layers.append(
               SimpleHGNConv(
                   edge_dim,
                   hidden_dim * heads[l - 1],
                   hidden_dim,
                   heads[l],
                   num_etypes,
                   feat_drop,
                   negative_slope,
                   residual,
                   self.activation,
                   beta=beta,
               )
           )
       # output projection
       self.hgn_layers.append(
           SimpleHGNConv(
               edge_dim,
               hidden_dim * heads,
               num_classes,
               1,
               num_etypes,
               feat_drop,
               negative_slope,
               residual,
               None,
               beta=beta,
           )
       )

   def forward(self, hg, h_dict):
       #输入 ： 异构图  和  dict类型的the feature dict of different node types
       #输出： 节点表征向量
       if hasattr(hg, 'ntypes'):  #检查异质图hg是否具有节点类型属性
           # full graph training,
           with hg.local_scope():
               hg.ndata['h'] = h_dict   #将初始节点特征赋予
               g = dgl.to_homogeneous(hg, ndata='h')  #转为同构图
               h = g.ndata['h']                        #初始嵌入维度
               for l in range(self.num_layers):  # noqa E741
                   h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                   h = h.flatten(1)
           #用于将同质图（Homogeneous Graph）中的特征转换为异质图（Heterogeneous Graph）的特征字典
           h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)


       return h_dict["Road"]





class SimpleHGNConv(nn.Module):
    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim     # 边的嵌入维度
        self.in_dim = in_dim         #输入特征的维度
        self.out_dim = out_dim       #输出特征的维度
        self.num_heads = num_heads   #几个头
        self.num_etypes = num_etypes #知识图谱中边类型的数目

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim))) #初始化边嵌入

        self.W = nn.Parameter(torch.FloatTensor(                              #实体可学习矩阵
            in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)    #关系可学习矩阵

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph  同构图
        h: tensor
            the original features of the graph  初始表征向量
        ntype: tensor
            the node type of the graph         节点类型1
        etype: tensor                          边类型
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """

        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)   #重塑为多头模式
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, #根据边类型etype选择相应的边特征。
                                                                         self.num_heads, self.edge_dim)

        row = g.edges()[0]     #获取头实体和尾实体
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]   #计算头实体、尾实体和边的注意力权重。
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)   #这里不是concat吗？？？？
        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']              #边残差注意力
            edge_attention = edge_attention * \
                             (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:                     #如果只有一个头，则调整注意力权重的形状。
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(fn.u_mul_e('emb', 'alpha', 'm'),
                         fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)

        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:                 # 是否使用残差
            res = self.residual(h)
            h_output += res
        if self.activation is not None:      #是否用激活
            h_output = self.activation(h_output)

        return h_output


class HAN(nn.Module):
    def __init__(self, etypes, in_size, hid_size, out_size, n_heads=2):
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, g_base, x):
        pass


class SeHGNN(nn.Module):
    def __init__(self, etypes, in_size, hid_size, out_size, n_heads=2):
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, g_base, x):
        pass



class R_GCN(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_size,
            hid_size,
            out_size,
            num_rels,
            regularizer="basis",
            num_bases=-1,
            dropout=0.0,
            self_loop=True,
            time=2880,
            poi_cate_num=12,
            fea_dimen=64
    ):
        super().__init__()

        if num_bases==-1:
            num_bases=num_rels


        self.conv1=dglnn.RelGraphConv(hid_size,hid_size,num_rels,regularizer,num_bases,self_loop=self_loop)

        self.conv2=dglnn.RelGraphConv(hid_size,hid_size,num_rels,regularizer,num_bases,self_loop=self_loop)


        self.dropout=nn.Dropout(dropout)


        self.poi_linear = nn.Linear(poi_cate_num, fea_dimen)
        self.time_linear = nn.Linear(time, fea_dimen*2)
        self.road_linear = nn.Linear(in_size, fea_dimen)

    def forward(self,g, g_base, x,road_ID,etype_base,road_embed,poi_distribution,time_distribution,road_feature):



        poi_feat = self.poi_linear(poi_distribution)
        time_feat = self.time_linear(time_distribution)
        road_feat = self.road_linear(road_feature)

        feat = th.cat([poi_feat, time_feat, road_feat], dim=1)
        input=feat

        h=self.conv1(g_base,input,etype_base)

        h=self.dropout(F.relu(h))
        h = self.conv2(g_base, h, etype_base)


        return h



class R_GCN2(nn.Module):
    def __init__(self, in_size, hid_size, out_size,num_rels,regularizer,num_bases,dropout,self_loop):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.RelGraphConv(in_size,hid_size,num_rels,regularizer,num_bases,self_loop=self_loop)
        )
        self.layers.append(  dglnn.RelGraphConv(hid_size,hid_size,out_size,regularizer,num_bases,self_loop=self_loop))
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, input,type):
        h = input
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(F.relu(h))
            h = layer(node, h,type)
        return h


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__( )
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias = False)
        self._norm_fact = 1 / sqrt(dim_k)   #缩放点积注意力机制

    def forward(self, x):
        # x: batch, n, dim_q

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n

        dist = torch.softmax(dist, dim = -1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att



class Linear2Infusion(nn.Module):
    def __init__(self, embeded):
        super(Linear2Infusion, self).__init__( )
        self.linear=nn.Linear(embeded,embeded)
        self.linear_KG=nn.Linear(embeded,embeded)
        self.gelu=nn.GELU()

    def forward(self,road_embedding,entity_embedding):
        road_embedding_=self.linear(road_embedding)
        entity_embedding_=self.linear_KG(entity_embedding)
        out_put=self.gelu(road_embedding_+entity_embedding_)

        return out_put




class Road_Emb(nn.Module):
    def __init__(
            self,
            hid_size,
            out_size,
            num_rels,
            feature_dim,
            time_dim=2880,
            poi_cate_num=12,
            regularizer="basis",
            num_bases=-1,
            dropout=0.0,
            self_loop=True,
    ):
        super().__init__()

        if num_bases==-1:
            num_bases=num_rels

        self.poi_linear = nn.Linear(poi_cate_num, 64)
        self.time_linear = nn.Linear(time_dim, 128)
        self.road_linear = nn.Linear(feature_dim, 64)
        self.adj_gcn=R_GCN2(hid_size,hid_size,out_size,num_rels,regularizer,num_bases,dropout,self_loop=self_loop)
        self.no_adj_gcn=R_GCN2(hid_size,hid_size,out_size,num_rels,regularizer,num_bases,dropout,self_loop=self_loop)
        self.line2fusion=Linear2Infusion(hid_size)




    def forward(self,adj_node,adj_type,no_adj_node,no_adj_type,poi_distribution,time_distribution,road_feature):


        poi_feat = self.poi_linear(poi_distribution)
        time_feat = self.time_linear(time_distribution)
        road_feat = self.road_linear(road_feature)

        feat = th.cat([poi_feat, time_feat, road_feat], dim=1)


        adj_road_emb=self.adj_gcn(adj_node,feat,adj_type)
        no_adj_road_emb=self.no_adj_gcn(no_adj_node,feat,no_adj_type)

        ##线性融合
        out= self.line2fusion(adj_road_emb,no_adj_road_emb)


        return out



class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layer_num, head_num):
        super().__init__()
        if layer_num == 0:
            self.gat_linear = nn.Linear(in_size, out_size)
        elif layer_num == 1:
            self.gat_layers = nn.ModuleList(
                [dglnn.GATConv(in_size, out_size, head_num, activation=F.elu)]
            )
            self.gat_linear = nn.Linear(out_size*head_num, out_size)
        else:
            self.gat_layers = nn.ModuleList()
            self.gat_layers.append(dglnn.GATConv(in_size, hid_size, head_num, activation=F.elu))
            for _ in range(layer_num - 2):
                self.gat_layers.append(dglnn.GATConv(hid_size*head_num, hid_size, head_num, activation=F.elu))
            self.gat_layers.append(dglnn.GATConv(hid_size*head_num, out_size, 1, activation=F.elu))
            self.gat_linear = nn.Identity()

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layer_num):
        super().__init__()
        self.layers = nn.ModuleList()
        if layer_num == 1:
            self.layers.append(
                dglnn.GraphConv(in_size, out_size)
            )
        else:
            # two-layer GCN
            self.layers.append(
                dglnn.GraphConv(in_size, hid_size, activation=F.relu)
            )
            for i in range(layer_num - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size))
            self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layer_num):
        super().__init__()
        self.layers = nn.ModuleList()
        if layer_num == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "gcn"))
        else:
            # two-layer GraphSAGE-mean
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
            for i in range(layer_num - 2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h