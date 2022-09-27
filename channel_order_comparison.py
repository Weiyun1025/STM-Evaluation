import torch
import torch.nn.functional as F


class Demo:
    def __init__(self):
        self.dim_head_qk = 32
        self.dim_head_v = 32
        self.block_size_ds = 7
        self.num_heads = 3
        self.block_size = 7
        self.halo_size = 3
        self.win_size = 13

    def channel_first_forward(self, q, kv):
        B, _, H, W = q.shape
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        # 12, 32, 7, 7, 8, 8
        # bsz x num_heads, dim_head, b_size, b_size, b_num, b_num
        q = q.reshape(
            -1, self.dim_head_qk,
            num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        # 12, 64, 49, 32
        # bsz x num_heads, b_num^2, b_size^2, dim_head
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)

        # 4, 192, 62, 62
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        # 12, 64, 169, 64
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        # 12, 64, 169, 32
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)

        return q, k, v

    def channel_last_forward(self, q, kv):
        B, H, W, _ = q.shape
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        # bsz, b_num, b_size, b_num, b_size, num_heads, head_dim
        q = q.reshape(B, num_h_blocks, self.block_size, num_w_blocks, self.block_size, self.num_heads, self.dim_head_qk)
        # bsz, num_heads, b_num, b_num, b_size, b_size, head_dim
        q = q.permute(0, 5, 1, 3, 2, 4, 6)
        # bsz x num_heads, b_num^2, b_size^2, dim_head
        q = q.reshape(B * self.num_heads, num_blocks, self.block_size * self.block_size, self.dim_head_qk)

        kv = F.pad(kv, (0, 0, self.halo_size, self.halo_size, self.halo_size, self.halo_size))
        kv = kv.unfold(1, self.win_size, self.block_size).unfold(2, self.win_size, self.block_size)
        kv = kv.reshape(B, num_blocks, self.num_heads, self.dim_head_qk + self.dim_head_v, self.win_size * self.win_size)
        kv = kv.permute(0, 2, 1, 4, 3).reshape(B * self.num_heads, num_blocks, self.win_size * self.win_size, self.dim_head_qk + self.dim_head_v)
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)

        return q, k, v

@profile
def speed_test():
    b, c, h, w = 4, 96, 56, 56
    q_in_1 = torch.randn(b, c, h, w)
    kv_in_1 = torch.randn(b, c + c, h, w)
    
    q_in_2 = torch.randn(b, h, w, c)
    kv_in_2 = torch.randn(b, h, w, c + c)

    model = Demo()

    for i in range(100):
        q1, k1, v1 = model.channel_first_forward(q_in_1, kv_in_1)
        q2, k2, v2 = model.channel_last_forward(q_in_2, kv_in_2)

        #assert torch.all(q1 == q2)
        #assert torch.all(k1 == k2)
        #assert torch.all(v1 == v2)


if __name__ == '__main__':
    speed_test()

# kernprof -l channel_order_comparison.py
# python -m line_profiler channel_order_comparison.py.lprof