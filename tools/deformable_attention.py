from types import MethodType
import math

import torch
import torch.nn.functional as F
from models.blocks.dcn_v3 import MSDeformAttnGrid_softmax


class Etmpy_MultiScaleDeformableAttnFunction(torch.autograd.Function):

    @staticmethod
    def symbolic(g, value, value_spatial_shapes, value_level_start_index,
                 sampling_locations, attention_weights, im2col_step):
        return g.op('com.microsoft::MultiscaleDeformableAttnPlugin_TRT', value,
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        '''
        no real mean,just for inference
        '''
        bs, _, mum_heads, embed_dims_num_heads = value.shape
        bs, num_queries, _, _, _, _ = sampling_locations.shape
        x = value.new_zeros(bs, num_queries, mum_heads *
                               embed_dims_num_heads).to(value.device)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        pass


def MSMHDA_onnx_export(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        device = query.device

        N, Len_q, C = query.shape
        pad_width = int(math.sqrt(self.n_points) // 2)
        H, W = input_spatial_shapes[:, 0], input_spatial_shapes[:, 1]
        H = H - 2 * pad_width
        W = W - 2 * pad_width

        assert input_flatten == None
        if input_flatten == None:
            input_flatten = query
        N, Len_in, C = input_flatten.shape

        query = query.permute(0, 2, 1).reshape(N, C, H, W)
        query = self.dw_conv(query).permute(
            0, 2, 3, 1).reshape(N, Len_q, C)

        # (N, Len-in, d_model)
        value = self.value_proj(input_flatten)

        # padding
        if self.padding:
            value = value.reshape(N, H, W, C)
            value = F.pad(
                value, [0, 0, pad_width, pad_width, pad_width, pad_width])
            value = value.reshape(N, -1, C)
            Len_in = value.size(1)  # update Len_in

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)
        value = value.reshape(N, Len_in, self.n_heads, int(
            self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).reshape(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            N, Len_q,
            self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).reshape(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points)
        self.grid = self.grid.to(device)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:  # 1-stage
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                (self.grid + sampling_offsets) * self.offsets_scaler / \
                offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))
        output = Etmpy_MultiScaleDeformableAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations.contiguous(), attention_weights, self.im2col_step)
        N, L, C = output.data.shape
        output = output.reshape(int(N), int(L), int(C))
        output = self.output_proj(output)
        return output


def register_defomable_attention(model):
    for moudle in model.modules():
        if isinstance(moudle, MSDeformAttnGrid_softmax):
            moudle.grid = moudle.grid.cuda()
            moudle.forward = MethodType(MSMHDA_onnx_export, moudle)