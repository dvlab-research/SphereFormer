import torch
import pointops
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_min, scatter_sum
import sys 
sys.path.append("..") 
import sptr

torch.manual_seed(1)

# M = 80000
N = 3500
n = 150
k = 65
# M = 80
# N = 5
hdim = 16
h = 6
L = 31
query = torch.rand(N, h, hdim).cuda()
table_q = torch.rand(L, h, hdim, 3).cuda()
key = torch.rand(N, h, hdim).cuda()
table_k = torch.rand(L, h, hdim, 3).cuda()

# index_q = torch.rand(M)
# index_q[index_q < 0] = 0
# index_q = (index_q*N).long().cuda()

# index_k = torch.rand(M)
# index_k[index_k < 0] = 0
# index_k = (index_k*N).long().cuda()

v2p_map = torch.randint(low=0, high=n, size=(N,)).cuda()
v2p_map, _ = v2p_map.sort()
counts = v2p_map.bincount()
M = (counts**2).sum().item()

# v2p_map, ctg_sort_idx = v2p_map.sort()
# n, k = p2v_map.shape
N = v2p_map.shape[0]
mask = torch.arange(k)[None].cuda().expand(n, -1) < counts[:, None] #[n, k]
to_add = torch.arange(k)[None].cuda().expand(n, -1)[mask]
v2p_map = v2p_map.long()
p2v_map = torch.zeros(n, k).long().cuda() #torch.zeros_like(p2v_map)
p2v_map[mask] = torch.arange(N).cuda()
ctg_index_1_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), (counts ** 2).cumsum(-1)], 0)[v2p_map] + to_add

rel_index = torch.rand(M, 3)
rel_index[rel_index < 0] = 0
rel_index = (rel_index*L).long().cuda()

index_q_counts = counts[v2p_map.long()] #[N, ]
index_q_offsets = index_q_counts.cumsum(-1) #[N, ]
index_q_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_q_offsets], 0) #[N+1]
n_max = p2v_map.shape[1]
index_q, index_k = pointops.precompute_index_pairs(p2v_map, counts, index_q_offsets)
index_q = index_q.long()
index_k = index_k.long()

# # rearrange index for acceleration
# index_q, indices = torch.sort(index_q) #[M,]
# index_k = index_k[indices] #[M,]
# rel_index = rel_index[indices]
# index_q_counts = index_q.bincount()

# print("index_q_counts.shape: ", index_q_counts.shape)

# n_max = index_q_counts.max()
# index_q_offsets = index_q_counts.cumsum(dim=-1) #[N]

# print("v1 index_q_offsets.shape: ", index_q_offsets.shape)

# index_q_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_q_offsets], 0) #[N+1]
        
# print("index_q[:100]: ", index_q[:100])
print("n_max: ", n_max)
print("index_q_offsets.shape: ", index_q_offsets.shape)
# input()

print("index_q_offsets[:100]: ", index_q_offsets[:100])
print("index_k[:20]: ", index_k[:20])

query.requires_grad = True
table_q.requires_grad = True
key.requires_grad = True
table_k.requires_grad = True

output1 = pointops.dot_prod_with_idx(query, index_q.int(), table_q, rel_index.int())
output2 = pointops.dot_prod_with_idx(key, index_k.int(), table_k, rel_index.int())
output = output1 + output2
loss = output.mean()
loss.backward()

print("output.shape: {}, output[:5,:10]: {}".format(output.shape, output[:5,:10]))
print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("table_q.grad[:5, :3, :5, :2]: ", table_q.grad[:5, :3, :5, :2])
print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
print("table_k.grad[:5, :3, :5, :2]: ", table_k.grad[:5, :3, :5, :2])
# input()

query_grad = query.grad.clone()
key_grad = key.grad.clone()
table_q_grad = table_q.grad.clone()
table_k_grad = table_k.grad.clone()

query.grad.zero_()
key.grad.zero_()
table_q.grad.zero_()
table_k.grad.zero_()


# print("query.is_contiguous(): ", query.is_contiguous())
# print("key.is_contiguous(): ", key.is_contiguous())
# print("index_q.is_contiguous(): ", index_q.is_contiguous())
# print("index_k.is_contiguous(): ", index_k.is_contiguous())
table_q = table_q.detach().permute(0,3,1,2).contiguous()
table_k = table_k.detach().permute(0,3,1,2).contiguous()
table_q.requires_grad = True
table_k.requires_grad = True
output_v2 = sptr.dot_prod_with_idx(query, index_q.int(), index_q_offsets.int(), n_max, key, ctg_index_1_offsets.int(), index_k.int(), table_q, table_k, rel_index.int())
# output_v2 = pointops.dot_prod_with_idx_v5(query, index_q_offsets.int(), n_max, key, index_k.int(), table_q, table_k, rel_index.int())
loss = output_v2.mean()
loss.backward()

table_q_grad2 = table_q.grad.clone().permute(0,2,3,1).contiguous()
table_k_grad2 = table_k.grad.clone().permute(0,2,3,1).contiguous()

print("output_v2.shape: {}, output_v2[:5,:10]: {}".format(output_v2.shape, output_v2[:5,:10]))
print("v2 query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("v2 table_q_grad2[:5, :3, :5, :2]: ", table_q_grad2[:5, :3, :5, :2])
print("v2 key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
print("v2 table_k_grad2[:5, :3, :5, :2]: ", table_k_grad2[:5, :3, :5, :2])
# input()

print("((output-output_v2)**2).max(): ", ((output-output_v2)**2).max())

print("((query.grad-query_grad)**2).max(): ", ((query.grad-query_grad)**2).max())

print("((key.grad-key_grad)**2).max(): ", ((key.grad-key_grad)**2).max())

print("((table_q_grad2-table_q_grad)**2).max(): ", ((table_q_grad2-table_q_grad)**2).max())

print("((table_k_grad2-table_k_grad)**2).max(): ", ((table_k_grad2-table_k_grad)**2).max())



