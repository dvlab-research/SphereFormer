import torch
import pointops
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_min, scatter_sum
import sys 
sys.path.append("..") 
import sptr

torch.manual_seed(2)

# M = 80000
N = 3500 #10 #5 #10 #3500
n = 150 #2 #150
# k = 5 #7 #65
hdim = 16
h = 6 #1 #6
L = 31 #2 #31
v = torch.rand(N, h, hdim).cuda()
# table = torch.rand(L, h, hdim, 3).cuda()

# index_0 = torch.rand(M)
# index_0[index_0 < 0] = 0
# index_0 = (index_0*N).long().cuda()

# index_1 = torch.rand(M)
# index_1[index_1 < 0] = 0
# index_1 = (index_1*N).long().cuda()


v2p_map = torch.randint(low=0, high=n, size=(N,)).cuda()
v2p_map, _ = v2p_map.sort()
counts = v2p_map.bincount()
M = (counts**2).sum().item()
k = counts.max().item()

print("counts: ", counts)

attn = torch.rand(M, h).cuda()

# v2p_map, ctg_sort_idx = v2p_map.sort()
# n, k = p2v_map.shape
# N = v2p_map.shape[0]
mask = torch.arange(k)[None].cuda().expand(n, -1) < counts[:, None] #[n, k]
to_add = torch.arange(k)[None].cuda().expand(n, -1)[mask]
v2p_map = v2p_map.long()
p2v_map = torch.zeros(n, k).long().cuda() #torch.zeros_like(p2v_map)
p2v_map[mask] = torch.arange(N).cuda()
ctg_index_1_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), (counts ** 2).cumsum(-1)], 0)[v2p_map] + to_add

print("M: ", M)
# print("counts[:5]: {}, v2p_map[:5]: {}, p2v_map[:5]: {}".format(counts[:5], v2p_map[:5], p2v_map[:5]))
# print("ctg_index_1_offsets[:50]: ", ctg_index_1_offsets[:50])

# print("ctg_index_1_offsets.max(): {}, ctg_index_1_offsets.min(): {}".format(ctg_index_1_offsets.max(), ctg_index_1_offsets.min()))
print("ctg_index_1_offsets: ", ctg_index_1_offsets)

# rel_index = torch.rand(M, 3)
# rel_index[rel_index < 0] = 0
# rel_index = (rel_index*L).long().cuda()

# # print("rel_index.min(): {}, rel_index.max(): {}".format(
# #     rel_index.min(), rel_index.max()
# # ))

# print("rel_index: ", rel_index)

index_0_counts = counts[v2p_map.long()] #[N, ]
index_0_offsets = index_0_counts.cumsum(-1) #[N, ]
index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0) #[N+1]
n_max = p2v_map.shape[1]
index_0, index_1 = pointops.precompute_index_pairs(p2v_map, counts, index_0_offsets)
index_0 = index_0.long()
index_1 = index_1.long()

print("index_0: {}".format(index_0))
print("index_1: {}".format(index_1))

# print("index_0.max(): {}, index_0.min(): {}".format(index_0.max(), index_0.min()))

# print("index_1.max(): {}, index_1.min(): {}".format(index_1.max(), index_1.min()))

# input()

# # rearrange index for acceleration
# index_0, indices = torch.sort(index_0) #[M,]
# index_1 = index_1[indices] #[M,]
# rel_index = rel_index[indices]
# index_0_counts = index_0.bincount()

print("index_0_counts.shape: ", index_0_counts.shape)

# n_max = index_0_counts.max()
# index_0_offsets = index_0_counts.cumsum(dim=-1) #[N]

# print("v1 index_0_offsets.shape: ", index_0_offsets.shape)

# index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0) #[N+1]


attn.requires_grad = True
v.requires_grad = True
# table.requires_grad = True


# output = pointops.attention_step2_with_rel_pos_value(attn, v, index_0.int(), index_1.int(), table, rel_index.int())
output = pointops.attention_step2(attn, v, index_0.int(), index_1.int())
loss = output.mean()
loss.backward()

# print("output.shape: {}, output[:5,:10,:5]: {}".format(output.shape, output[:5,:10, :5]))
# print("attn.grad[:5, :3]: ", attn.grad[:5, :3])
# print("v.grad[:5, :3, :5]: ", v.grad[:5, :3, :5])
# print("table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
# # input()

attn_grad = attn.grad.clone()
v_grad = v.grad.clone()
# table_grad = table.grad.clone()

attn.grad.zero_()
v.grad.zero_()
# table.grad.zero_()

# print("query.is_contiguous(): ", query.is_contiguous())
# print("key.is_contiguous(): ", key.is_contiguous())
# print("index_0.is_contiguous(): ", index_0.is_contiguous())
# print("index_1.is_contiguous(): ", index_1.is_contiguous())

# output_v2 = pointops.attention_step2_with_rel_pos_value_v7(attn, v, index_0.int(), index_0_offsets.int(), n_max, index_1.int(), ctg_index_1_offsets.int(), table, rel_index.int())
# output_v2 = pointops.attention_step2_with_rel_pos_value_v4(attn, v, index_0_offsets.int(), n_max, index_1.int(), table, rel_index.int())
output_v2 = sptr.attention_step2(attn, v, index_0.int(), index_0_offsets.int(), n_max, index_1.int(), ctg_index_1_offsets.int())
loss = output_v2.mean()
loss.backward()

# print("output_v2.shape: {}, output_v2[:5,:10,:5]: {}".format(output_v2.shape, output_v2[:5,:10,:5]))
# print("v2 attn.grad[:5, :3]: ", attn.grad[:5, :3])
# print("v2 v.grad[:5, :3, :5]: ", v.grad[:5, :3, :5])
# print("v2 table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
# # input()

print("((output-output_v2)**2).max(): ", ((output-output_v2)**2).max())

print("((attn_grad-attn.grad)**2).max(): ", ((attn_grad-attn.grad)**2).max())

print("((v_grad-v.grad)**2).max(): ", ((v_grad-v.grad)**2).max())

# print("((table_grad-table.grad)**2).max(): ", ((table_grad-table.grad)**2).max())

# print("torch.max((attn_flat-attn_flat_v2)**2): ", torch.max((attn_flat-attn_flat_v2)**2))

