import torch
import pointops
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_min, scatter_sum
import sys 
sys.path.append("..") 
import sptr

torch.manual_seed(1)

# M = 800000
N = 35000
n = 1500
k = 100
C = 96
h = 6
query = torch.rand(N, h, C//h).cuda()
key = torch.rand(N, h, C//h).cuda()

v2p_map = torch.randint(low=0, high=n, size=(N,)).cuda()
v2p_map, _ = v2p_map.sort()
counts = v2p_map.bincount()
M = (counts**2).sum().item()

N = v2p_map.shape[0]
mask = torch.arange(k)[None].cuda().expand(n, -1) < counts[:, None] #[n, k]
to_add = torch.arange(k)[None].cuda().expand(n, -1)[mask]
v2p_map = v2p_map.long()
p2v_map = torch.zeros(n, k).long().cuda() #torch.zeros_like(p2v_map)
p2v_map[mask] = torch.arange(N).cuda()
ctg_index_1_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), (counts ** 2).cumsum(-1)], 0)[v2p_map] + to_add

index_0_counts = counts[v2p_map.long()] #[N, ]
index_0_offsets = index_0_counts.cumsum(-1) #[N, ]
index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0) #[N+1]
n_max = p2v_map.shape[1]
index_0, index_1 = pointops.precompute_index_pairs(p2v_map, counts, index_0_offsets)
index_0 = index_0.long()
index_1 = index_1.long()

# index_0 = torch.rand(M)
# index_0[index_0 < 0] = 0
# index_0 = (index_0*N).long().cuda()

# index_1 = torch.rand(M)
# index_1[index_1 < 0] = 0
# index_1 = (index_1*N).long().cuda()

query.requires_grad = True
key.requires_grad = True

# # rearrange index for acceleration
# index_0, indices = torch.sort(index_0) #[M,]
# index_1 = index_1[indices] #[M,]
# index_0_counts = index_0.bincount()
# n_max = index_0_counts.max()
# index_0_offsets = index_0_counts.cumsum(dim=-1) #[N]
# index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0) #[N+1]

attn_flat = pointops.attention_step1(query.float(), key.float(), index_0.int(), index_1.int())
loss = attn_flat.sum()
loss.backward()
print("attn_flat.shape: {}, attn_flat[:20,:10]: {}".format(attn_flat.shape, attn_flat[:20,:10]))
print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
# input()

query_grad = query.grad.clone()
key_grad = key.grad.clone()

query.grad.zero_()
key.grad.zero_()

# # print("index_0[:100]: ", index_0[:100])
# print("n_max: ", n_max)
# print("index_0_offsets.shape: ", index_0_offsets.shape)
# # input()

# print("index_0_offsets[:100]: ", index_0_offsets[:100])
# print("index_1[:20]: ", index_1[:20])


# attn_flat = pointops.attention_step1(query.float(), key.float(), index_0.int(), index_1.int())
# loss = attn_flat.sum()
# loss.backward()
# # attn_flat = pointops.attention_step1(query.float(), key.float(), index_0.int(), index_1.int())
# # loss = attn_flat.sum()
# # loss.backward()
# print("attn_flat.shape: {}, attn_flat[:20,:10]: {}".format(attn_flat.shape, attn_flat[:20,:10]))
# print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
# print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
# input()

print("query.is_contiguous(): ", query.is_contiguous())
print("key.is_contiguous(): ", key.is_contiguous())
print("index_0.is_contiguous(): ", index_0.is_contiguous())
print("index_1.is_contiguous(): ", index_1.is_contiguous())

attn_flat_v2 = sptr.attention_step1(query.float(), key.float(), index_0.int(), index_0_offsets.int(), index_1.int(), ctg_index_1_offsets.int(), n_max)
loss = attn_flat_v2.sum()
loss.backward()

# attn_flat_v2 = pointops.attention_step1_v2(query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max)
# loss = attn_flat_v2.sum()
# loss.backward()

print("attn_flat_v2.shape: {}, attn_flat_v2[:20,:10]: {}".format(attn_flat_v2.shape, attn_flat_v2[:20,:10]))
print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
# input()

# mask = attn_flat_v2.sum(-1) != 0
# print("mask.sum(): ", mask.sum())
# print("attn_flat_v2[mask] - attn_flat[mask]: ", ((attn_flat_v2[mask] - attn_flat[mask])**2).max())


print("((attn_flat-attn_flat_v2).abs()).max(): ", ((attn_flat-attn_flat_v2).abs()).max())

print("(query.grad-query_grad).abs().max(): ", (query.grad-query_grad).abs().max())
print("(key.grad-key_grad).abs().max(): ", (key.grad-key_grad).abs().max())

# selected = 10000
# print("torch.max((attn_flat[:selected]-attn_flat_v2[:selected])**2, 0): ", torch.max((attn_flat[:selected]-attn_flat_v2[:selected])**2, 0))

