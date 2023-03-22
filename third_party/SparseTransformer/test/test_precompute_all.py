import torch
import pointops
import sys 
sys.path.append("..") 
import sptr

torch.manual_seed(1)

v2p_map = torch.IntTensor([
    1, 0, 0, 2, 0, 2, 2, 1, 2, 2, 2
]).cuda()

p2v_map = torch.IntTensor([
    [1, 2, 4, 0, 0, 0],
    [0, 7, 0, 0, 0, 0],
    [5, 6, 3, 9, 8, 10]
]).cuda()

counts = torch.IntTensor([3, 2, 6]).cuda()

# index_0_counts = counts[v2p_map.long()] #[N, ]
# index_0_offsets = index_0_counts.cumsum(-1) #[N, ]
# index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0) #[N+1]

# N = v2p_map.shape[0]

print("v2p_map.shape: {}, p2v_map.shape: {}".format(v2p_map.shape, p2v_map.shape))

v2p_map, ctg_sort_idx = v2p_map.sort()
n, k = p2v_map.shape
N = v2p_map.shape[0]
mask = torch.arange(k)[None].cuda().expand(n, -1) < counts[:, None] #[n, k]
to_add = torch.arange(k)[None].cuda().expand(n, -1)[mask]
v2p_map = v2p_map.long()
p2v_map = torch.zeros_like(p2v_map)
p2v_map[mask] = torch.arange(N).int().cuda()
ctg_index_1_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), (counts ** 2).cumsum(-1)], 0)[v2p_map] + to_add

index_params = (ctg_index_1_offsets, ctg_sort_idx)
index_0_counts = counts[v2p_map.long()] #[N, ]
index_0_offsets = index_0_counts.cumsum(-1) #[N, ]
index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0) #[N+1]
n_max = p2v_map.shape[1]
index_0, index_1 = pointops.precompute_index_pairs(p2v_map, counts, index_0_offsets)
index_0 = index_0.long()
index_1 = index_1.long()


# index_0_offsets_, index_1_offsets_ = pointops.precompute_offsets(N, p2v_map.shape[0], p2v_map.shape[1], counts)

# # assert (index_0_offsets_ == index_0_offsets).all()
# print("index_0_offsets: ", index_0_offsets)
# print("index_0_offsets_: ", index_0_offsets_)
# print("ctg_index_1_offsets: ", ctg_index_1_offsets)
# print("index_1_offsets_: ", index_1_offsets_)

# index_0, index_1 = pointops.precompute_index_pairs(p2v_map, counts, index_0_offsets)

# print("index_0: ", index_0)
# print("index_1: ", index_1)

# print("index_0.shape: ", index_0.shape)


index_0_offsets_, index_1_offsets_, index_0_, index_1_ = sptr.precompute_all(N, p2v_map.shape[0], p2v_map.shape[1], counts)

assert (index_0_offsets_ == index_0_offsets).all()
assert (index_1_offsets_ == ctg_index_1_offsets).all()
assert (index_0_ == index_0).all()
assert (index_1_ == index_1).all()

print("index_0_offsets: ", index_0_offsets)
print("index_0_offsets_: ", index_0_offsets_)
print("ctg_index_1_offsets: ", ctg_index_1_offsets)
print("index_1_offsets_: ", index_1_offsets_)

print("index_0_offsets.shape: {}, index_0_offsets_.shape: {}".format(index_0_offsets.shape, index_0_offsets_.shape))

print("ctg_index_1_offsets.shape: {}, index_1_offsets_.shape: {}".format(ctg_index_1_offsets.shape, index_1_offsets_.shape))

# print("index_0: ", index_0)
# print("index_0_: ", index_0_)
# print("index_1: ", index_1)
# print("index_1_: ", index_1_)

# print("index_0.shape: ", index_0.shape)
