from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops2_cuda as pointops_cuda
import time
# from util.common_util import Timer

DEBUG = False

class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

furthestsampling = FurthestSampling.apply


class PrecomputeIndexPairs(Function):
    @staticmethod
    def forward(ctx, p2v_map, counts, index_0_offsets):
        """
        input: p2v_map: (n, k), counts: (n), index_0_offsets: (N+1)
        output: index_0, index_1: [M]
        """
        assert p2v_map.is_contiguous() and counts.is_contiguous() and index_0_offsets.is_contiguous()
        n, k = p2v_map.shape
        N = index_0_offsets.shape[0] - 1
        M = (counts ** 2).sum().item()
        
        # assert k <= 1024
        assert k <= 4096

        index_0 = torch.cuda.IntTensor(M).zero_()
        index_1 = torch.cuda.IntTensor(M).zero_()
        pointops_cuda.precompute_index_pairs_cuda(N, n, k, p2v_map.int(), counts.int(), index_0_offsets.int(), index_0, index_1)
        return index_0, index_1

precompute_index_pairs = PrecomputeIndexPairs.apply

class PrecomputeOffsets(Function):
    @staticmethod
    def forward(ctx, N, n, n_max, counts):
        """
        input: p2v_map: (n, k), counts: (n), index_0_offsets: (N+1)
        output: index_0, index_1: [M]
        """
        assert counts.is_contiguous()
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(-1)], 0)
        sq_offsets = torch.cat([counts.new_zeros(1), (counts**2).cumsum(-1)], 0)

        print("counts: ", counts)
        print("offsets: ", offsets)
        print("sq_offsets: ", sq_offsets)

        assert n_max <= 1024
        index_0_offsets = torch.cuda.IntTensor(N).zero_()
        index_1_offsets = torch.cuda.IntTensor(N).zero_()

        pointops_cuda.precompute_offsets_cuda(N, n, n_max, counts.int(), offsets.int(), sq_offsets.int(), index_0_offsets, index_1_offsets)
        return index_0_offsets, index_1_offsets

precompute_offsets = PrecomputeOffsets.apply


class PrecomputeAll(Function):
    @staticmethod
    def forward(ctx, N, n, n_max, counts):
        """
        input: p2v_map: (n, k), counts: (n), index_0_offsets: (N+1)
        output: index_0, index_1: [M]
        """
        assert counts.is_contiguous()
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(-1)], 0)
        sq_offsets = torch.cat([counts.new_zeros(1), (counts**2).cumsum(-1)], 0)
        M = sq_offsets[-1].item()

        # print("counts: ", counts)
        # print("offsets: ", offsets)
        # print("sq_offsets: ", sq_offsets)
        # print("M: ", M)

        assert n_max <= 1024
        index_0_offsets = torch.cuda.IntTensor(N).zero_()
        index_1_offsets = torch.cuda.IntTensor(N).zero_()
        index_0 = torch.cuda.IntTensor(M).zero_()
        index_1 = torch.cuda.IntTensor(M).zero_()

        pointops_cuda.precompute_all_cuda(N, n, n_max, counts.int(), offsets.int(), sq_offsets.int(), index_0_offsets, index_1_offsets, index_0, index_1)
        index_0_offsets = torch.cat([index_0_offsets, torch.tensor([M]).cuda()], 0)
        return index_0_offsets, index_1_offsets, index_0, index_1

precompute_all = PrecomputeAll.apply


class PrecomputeIndexPairsQK(Function):
    @staticmethod
    def forward(ctx, p2v_map, p2v_map2, counts, counts2):
        """
        input: p2v_map: (n, k), p2v_map2: (n, k2), counts: (n), counts2: (n)
        output: index_0, index_1: [M]
        """
        assert p2v_map.is_contiguous() and p2v_map2.is_contiguous() and counts.is_contiguous() and counts2.is_contiguous()
        n, k = p2v_map.shape
        k2 = p2v_map2.shape[1]
        counts_out = counts * counts2
        out_offsets = torch.cat([torch.tensor([0]).cuda().long(), counts_out], 0).cumsum(-1)
        M = counts_out.sum().item()
        N = counts.sum().item()

        # assert k <= 1024
        # assert k <= 4096

        index_0 = torch.cuda.IntTensor(M).zero_()
        index_1 = torch.cuda.IntTensor(M).zero_()
        pointops_cuda.precompute_index_pairs_q_k_cuda(N, n, k, k2, p2v_map.int(), p2v_map2.int(), counts.int(), counts2.int(), out_offsets.int(), index_0, index_1)
        return index_0, index_1

precompute_index_pairs_q_k = PrecomputeIndexPairsQK.apply

class PrecomputeIndexPairsV2(Function):
    @staticmethod
    def forward(ctx, p2v_map_down, counts_down, p2v_map, counts, index_0_offsets):
        """
        input: p2v_map_down: (n, k), counts_down: (n), counts: (n), index_0_offsets: (N+1)
        output: index_0, index_1: [M]
        """
        assert p2v_map_down.is_contiguous() and counts_down.is_contiguous() and p2v_map.is_contiguous() and counts.is_contiguous() and index_0_offsets.is_contiguous()
        n, k = p2v_map.shape
        k_down = p2v_map_down.shape[1]
        N = index_0_offsets.shape[0] - 1
        # M = (counts ** 2).sum().item()
        M = (counts_down * counts).sum().item()

        # assert k_down <= 1024
        assert k_down <= 4096

        index_0 = torch.cuda.IntTensor(M).zero_()
        index_1 = torch.cuda.IntTensor(M).zero_()
        pointops_cuda.precompute_index_pairs_cuda_v2(N, n, k_down, k, p2v_map_down.int(), counts_down.int(), p2v_map.int(), counts.int(), index_0_offsets.int(), index_0, index_1)
        return index_0, index_1

precompute_index_pairs_v2 = PrecomputeIndexPairsV2.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)

knnquery = KNNQuery.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None

grouping = Grouping.apply

class AttentionStep1(Function):
    @staticmethod
    def forward(ctx, q, k, index0, index1):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0.is_contiguous() and index1.is_contiguous()

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index0.shape[0]
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.attention_step1_forward_cuda(N_k, M, h, C, q, k, index0, index1, output)
        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.save_for_backward(q, k, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        q, k, index0, index1 = ctx.saved_tensors
        M, h = grad_output.shape
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step1_backward_cuda(N_q, M, h, C, grad_output, index0, index1, q, k, grad_q, grad_k)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v7: {}".format(end - start))
        # # input()
        
        return grad_q, grad_k, None, None

attention_step1 = AttentionStep1.apply

class AttentionStep1_v2(Function):
    @staticmethod
    def forward(ctx, q, k, index1, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous()
        # assert n_max <= 1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)

        assert C <= 1024

        output = torch.cuda.FloatTensor(M, h).zero_()

        if n_max <= 1024:
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()

            pointops_cuda.attention_step1_forward_cuda_v2(N_k, M, h, C, n_max, q, k, index0_offsets, index1, output)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step1_v2: forward time: {}s, M: {}".format(time.time() - t, M))
        else:

            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()

            pointops_cuda.attention_step1_forward_cuda_v6(N_k, M, h, C, n_max, q, k, index0_offsets, index1, output)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step1_v6: forward time: {}s, M: {}".format(time.time() - t, M))

        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0_offsets, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0_offsets, index1 = ctx.saved_tensors
        M, h = grad_output.shape
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        if n_max <= 1024:
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
            
            pointops_cuda.attention_step1_backward_cuda_v2(N_q, M, h, C, n_max, grad_output, index0_offsets, index1, q, k, grad_q, grad_k)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step1_v2: backward time: {}s, M: {}".format(time.time() - t, M))
                # input()
        else:
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
            
            pointops_cuda.attention_step1_backward_cuda_v6(N_q, M, h, C, n_max, grad_output, index0_offsets, index1, q, k, grad_q, grad_k)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step1_v6: backward time: {}s, M: {}".format(time.time() - t, M))
                # input()

        return grad_q, grad_k, None, None, None

attention_step1_v2 = AttentionStep1_v2.apply

class AttentionStep1_v7(Function):
    @staticmethod
    def forward(ctx, q, k, index1, index1_offsets, index0, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous()
        # assert n_max <= 1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)

        # assert C <= 1024
        n_h = 512 // C_div_h if C > 512 else h
        assert h % n_h == 0

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()

        k = k.permute(1,2,0).contiguous()


        pointops_cuda.attention_step1_forward_cuda_v7(N_k, M, h, C, n_max, q, k, index0_offsets, index1, output)
        
        k = k.permute(2,0,1).contiguous()
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step1_v7: forward time: {}s, M: {}".format(time.time() - t, M))

        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0, index0_offsets, index1, index1_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0, index0_offsets, index1, index1_offsets = ctx.saved_tensors
        M, h = grad_output.shape
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step1_backward_cuda_v7(N_q, M, h, C, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step1_v7: backward time: {}s, M: {}".format(time.time() - t, M))
            # input()

        # print("grad_q.shape: ", grad_q.shape)

        return grad_q, grad_k, None, None, None, None, None

attention_step1_v7 = AttentionStep1_v7.apply


class AttentionStep1_v8(Function):
    @staticmethod
    def forward(ctx, q, k, index1, index1_offsets, index0, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous()
        # assert n_max <= 1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)

        # assert C <= 1024
        n_h = 512 // C_div_h if C > 512 else h
        assert h % n_h == 0

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()

        k_transpose = k.permute(1,2,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        pointops_cuda.attention_step1_forward_cuda_v7(N_k, M, h, C, n_max, q, k_transpose, index0_offsets, index1, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step1_v8: forward time: {}s, M: {}".format(time.time() - t, M))

        del k_transpose
        # k = k.permute(2,0,1).contiguous()
        output = output.permute(1,0).contiguous()

        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0, index0_offsets, index1, index1_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0, index0_offsets, index1, index1_offsets = ctx.saved_tensors
        M, h = grad_output.shape
        
        # grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step1_backward_cuda_v7(N_q, M, h, C, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step1_v8: backward time: {}s, M: {}".format(time.time() - t, M))
            # input()

        # print("grad_q.shape: ", grad_q.shape)

        return grad_q, grad_k, None, None, None, None, None

attention_step1_v8 = AttentionStep1_v8.apply


class AttentionStep1_v9(Function):
    @staticmethod
    def forward(ctx, q, k, index1, index1_offsets, index0, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous()
        # assert n_max <= 1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)

        # assert C <= 1024
        n_h = 512 // C_div_h if C > 512 else h
        assert h % n_h == 0

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()

        q = q.permute(1,2,0).contiguous()
        k = k.permute(1,2,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        pointops_cuda.attention_step1_forward_cuda_v9(N_k, M, h, C, n_max, q, k, index0, index1, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step1_v9: forward time: {}s, M: {}".format(time.time() - t, M))

        q = q.permute(2,0,1).contiguous()
        k = k.permute(2,0,1).contiguous()
        output = output.permute(1,0).contiguous()

        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0, index0_offsets, index1, index1_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0, index0_offsets, index1, index1_offsets = ctx.saved_tensors
        M, h = grad_output.shape
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step1_backward_cuda_v7(N_q, M, h, C, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step1_v7: backward time: {}s, M: {}".format(time.time() - t, M))
            # input()

        # print("grad_q.shape: ", grad_q.shape)

        return grad_q, grad_k, None, None, None, None, None

attention_step1_v9 = AttentionStep1_v9.apply

class AttentionStep1_v3(Function):
    @staticmethod
    def forward(ctx, q, k, index0, index1, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index0.is_contiguous() and index1.is_contiguous()
        assert q.is_cuda and k.is_cuda and index0_offsets.is_cuda and index0.is_cuda and index1.is_cuda
        assert n_max <= 1024

        N_q, h, C_div_h = q.shape
        # N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)
        N = index0_offsets.shape[0] - 1

        # print("k.shape: {}, index0_offsets.shape: {}".format(k.shape, index0_offsets.shape))

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.attention_step1_forward_cuda_v3(N, M, h, C, n_max, q.float(), k.float(), index0_offsets.int(), index0.int(), index1.int(), output)
        ctx.N_q = N_q
        # ctx.N_k = N_k
        ctx.N = N
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0_offsets, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N = ctx.N
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0_offsets, index0, index1 = ctx.saved_tensors
        M, h = grad_output.shape
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        assert q.is_cuda and k.is_cuda and index0_offsets.is_cuda and index0.is_cuda and index1.is_cuda and grad_output.is_cuda

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step1_backward_cuda_v3(N, M, h, C, n_max, grad_output.float(), index0_offsets.int(), index0.int(), index1.int(), q.float(), k.float(), grad_q, grad_k)
        
        # print("grad_q[0,0,0]: ", grad_q[0,0,0])

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v7: {}, M: {}".format(end - start, M))
        # input()
        
        return grad_q, grad_k, None, None, None, None

attention_step1_v3 = AttentionStep1_v3.apply


class AttentionStep1_v4(Function):
    @staticmethod
    def forward(ctx, q, k, index1, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous()
        assert n_max <= 256 #1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)

        print("index0_offsets: {}, N_k: {}, index0_offsets.shape: {}".format(
            index0_offsets, N_k, index0_offsets.shape
        ))
        input()

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.attention_step1_forward_cuda_v4(N_k, M, h, C, n_max, q, k, index0_offsets, index1, output)
        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0_offsets, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0_offsets, index1 = ctx.saved_tensors
        M, h = grad_output.shape
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step1_backward_cuda_v4(N_q, M, h, C, n_max, grad_output, index0_offsets, index1, q, k, grad_q, grad_k)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v8: {}, M: {}".format(end - start, M))
        # input()
        
        return grad_q, grad_k, None, None, None

attention_step1_v4 = AttentionStep1_v4.apply


class AttentionStep1_v5(Function):
    @staticmethod
    def forward(ctx, q, k, p2v_map, counts, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert q.is_contiguous() and k.is_contiguous() and p2v_map.is_contiguous() and counts.is_contiguous()
        assert n_max <= 256 #1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        # M = index1.shape[0]
        C = int(C_div_h * h)

        # print("index0_offsets: {}, N_k: {}, index0_offsets.shape: {}".format(
        #     index0_offsets, N_k, index0_offsets.shape
        # ))
        # input()

        n = p2v_map.shape[0]
        offset = (counts ** 2).cumsum(-1)
        offset = torch.cat([torch.IntTensor([0]).cuda(), offset], 0)
        # print("offset: ", offset)
        M = offset[-1].item()
        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.attention_step1_forward_cuda_v5(N_k, M, h, C, n, n_max, q, k, p2v_map, counts, offset.int(), output)
        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, p2v_map, counts, offset)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        
        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0_offsets, index1 = ctx.saved_tensors
        M, h = grad_output.shape
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert q.is_contiguous() and k.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C//h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step1_backward_cuda_v5(N_q, M, h, C, n_max, grad_output, index0_offsets, index1, q, k, grad_q, grad_k)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v8: {}, M: {}".format(end - start, M))
        # input()
        
        return grad_q, grad_k, None, None, None

attention_step1_v5 = AttentionStep1_v5.apply


class AttentionStep2(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index1):
        """
        input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index1.is_contiguous()

        M, h = attn.shape
        N_q = index0.max().item() + 1
        N_v, h, C_div_h = v.shape
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(N_q, h, C//h).zero_()
        pointops_cuda.attention_step2_forward_cuda(N_q, M, h, C, attn, v, index0, index1, output)
        ctx.M = M

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.save_for_backward(attn, v, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        M = ctx.M
        attn, v, index0, index1 = ctx.saved_tensors
        N_v = v.shape[0]
        N_q, h, C_div_h = grad_output.shape
        C = h * C_div_h
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_v, h, C//h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step2_backward_cuda(N_q, M, h, C, grad_output, index0, index1, attn, v, grad_attn, grad_v)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v8: {}".format(end - start))
        # # input()
        
        return grad_attn, grad_v, None, None

attention_step2 = AttentionStep2.apply


class AttentionStep2_v2(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index1):
        """
        input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)
        output: output: [L, h, C//h]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index1.is_contiguous()
        
        L = int(index0.max().item()) + 1

        M, h = attn.shape
        N, h, C_div_h = v.shape
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(L, h, C//h).zero_()
        pointops_cuda.attention_step2_forward_cuda(N, M, h, C, attn, v, index0, index1, output)
        ctx.M = M

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.save_for_backward(attn, v, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (L, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        M = ctx.M
        attn, v, index0, index1 = ctx.saved_tensors
        L, h, C_div_h = grad_output.shape
        N = v.shape[0]
        C = h * C_div_h
        
        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N, h, C//h).zero_()

        pointops_cuda.attention_step2_backward_cuda(N, M, h, C, grad_output, index0, index1, attn, v, grad_attn, grad_v)
        return grad_attn, grad_v, None, None

attention_step2_v2 = AttentionStep2_v2.apply

class AttentionStep2_v7(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        # L = table.shape[0]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert attn.shape[1] == h
        assert hdim == 16
        # assert L <= 50

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        # table = table.permute(0,3,1,2).contiguous() #[L, 3, h, hdim]

        # pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        pointops_cuda.attention_step2_forward_cuda_v7(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_v7: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # print("attn[:5,:5]: ", attn[:5, :5])
        # table = table.permute(0,2,3,1).contiguous()

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        # L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        # grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        # grad_table = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()


        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        # table = table.permute(1,2,3,0).contiguous()
        v = v.permute(1,2,0).contiguous()
        # rel_idx = rel_idx.permute(1,0).contiguous()

        # print("grad_output.shape: {}, index0_offsets.shape: {}, index1.shape: {}, index1_offsets.shape: {}, attn.shape: {}, v.shape: {}, table.shape: {}, rel_idx.shape: {}, grad_attn.shape: {}, grad_v,.shape: {} grad_table.shape: {}".format(
        #     grad_output.shape, index0_offsets.shape, index1.shape, index1_offsets.shape, attn.shape, v.shape, table.shape, rel_idx.shape, grad_attn.shape, grad_v.shape, grad_table.shape
        # ))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_backward_cuda_v7(N, M, h, hdim, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, grad_attn, grad_v)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_v7: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # grad_table = grad_table.permute(0,2,3,1).contiguous()

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, None

attention_step2_v7 = AttentionStep2_v7.apply

class DotProdWithIdx(Function):
    @staticmethod
    def forward(ctx, q, index, table, rel_idx):
        """
        input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index.shape[0]

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        ctx.save_for_backward(q, index, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index, table, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda(N, M, h, hdim, grad_output, q, index, table, rel_idx, grad_q, grad_table)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v9: {}".format(end - start))
        # # input()
        
        return grad_q, None, grad_table, None

dot_prod_with_idx = DotProdWithIdx.apply

class DotProdWithIdx_v2(Function):
    @staticmethod
    def forward(ctx, q, index_q, k, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_q.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L and index_k.shape[0] == M

        # obtain the mapping from block_idx to m_idx
        rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        sorted_values, sort_indices = torch.sort(rel_idx_merge)
        _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        n_max = counts.max()
        T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        output = torch.cuda.FloatTensor(M, h).zero_()
        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v2(N, M, h, hdim, n_max, T, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets.int(), sort_indices.int(), output)
        
        ctx.n_max = n_max
        ctx.T = T
        ctx.save_for_backward(q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        T, n_max = ctx.T, ctx.n_max
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and rel_idx_offsets.is_contiguous() and sort_indices.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_k = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v2(N, M, h, hdim, n_max, T, grad_output, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets.int(), sort_indices.int(), grad_q, grad_k, grad_table_q, grad_table_k)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v9: {}".format(end - start))
        # # input()
        return grad_q, None, grad_k, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v2 = DotProdWithIdx_v2.apply


class DotProdWithIdx_v3(Function):
    @staticmethod
    def forward(ctx, q, index_q_offsets, n_max, k, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        output = torch.cuda.FloatTensor(M, h).zero_()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v3: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q_offsets, k, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, k, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v3(N, M, h, hdim, n_max, grad_output, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v3: backward time: {}s, M: {}".format(time.time() - t, M))
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, grad_k, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v3 = DotProdWithIdx_v3.apply


class DotProdWithIdx_v7(Function):
    @staticmethod
    def forward(ctx, q, index_q_offsets, n_max, k, index_k_offsets, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 50

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()
        
        k = k.permute(1,2,0).contiguous()
        table_q = table_q.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        table_k = table_k.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        rel_idx = rel_idx.permute(1,0).contiguous()

        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        # pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v7(N, M, h, hdim, n_max, L, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        
        k = k.permute(2,0,1).contiguous()
        table_q = table_q.permute(3,0,1,2).contiguous()
        table_k = table_k.permute(3,0,1,2).contiguous() #[L, h, hdim, 3]
        rel_idx = rel_idx.permute(1,0).contiguous()
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v7: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 50
        assert 512 % hdim == 0

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()

        table_q = table_q.permute(0,3,1,2).contiguous()
        table_k = table_k.permute(0,3,1,2).contiguous()  #[L, h, hdim, 3] -> [L, 3, h, hdim]
        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v7(N, M, h, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v7: backward time: {}s, M: {}".format(time.time() - t, M))
        
        grad_table_q = grad_table_q.permute(0,2,3,1).contiguous()
        grad_table_k = grad_table_k.permute(0,2,3,1).contiguous()

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, grad_k, None, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v7 = DotProdWithIdx_v7.apply

class DotProdWithIdx_v8(Function):
    @staticmethod
    def forward(ctx, q, index_q_offsets, n_max, k, index_k_offsets, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, 3, h, hdim), table_k: (L, 3, h, hdim), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 50

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()
        
        k_transpose = k.permute(1,2,0).contiguous()
        # table_q_transpose = table_q.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        # table_k_transpose = table_k.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        table_q_transpose = table_q.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        table_k_transpose = table_k.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        rel_idx_transpose = rel_idx.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        # pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v7(N, M, h, hdim, n_max, L, q, index_q_offsets, k_transpose, index_k, table_q_transpose, table_k_transpose, rel_idx_transpose, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v8: forward time: {}s, M: {}".format(time.time() - t, M))
        
        # k = k.permute(2,0,1).contiguous()
        # table_q = table_q.permute(3,0,1,2).contiguous()
        # table_k = table_k.permute(3,0,1,2).contiguous()
        # rel_idx = rel_idx.permute(1,0).contiguous()
        output = output.permute(1,0).contiguous()

        del k_transpose
        del table_q_transpose
        del table_k_transpose
        del rel_idx_transpose

        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        # grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 50
        assert 512 % hdim == 0

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()

        # table_q = table_q.permute(0,3,1,2).contiguous()
        # table_k = table_k.permute(0,3,1,2).contiguous()
        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v7(N, M, h, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v8: backward time: {}s, M: {}".format(time.time() - t, M))
        
        # grad_table_q = grad_table_q.permute(0,2,3,1).contiguous()
        # grad_table_k = grad_table_k.permute(0,2,3,1).contiguous()

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, grad_k, None, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v8 = DotProdWithIdx_v8.apply


class DotProdWithIdx_v9(Function):
    @staticmethod
    def forward(ctx, q, index_q, index_q_offsets, n_max, k, index_k_offsets, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 50

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()
        
        q = q.permute(1,2,0).contiguous() #[h, hdim, L]
        k = k.permute(1,2,0).contiguous() #[h, hdim, L]
        table_q = table_q.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        table_k = table_k.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        rel_idx = rel_idx.permute(1,0).contiguous() #[3, M]

        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        # pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v9(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        
        q = q.permute(2,0,1).contiguous()
        k = k.permute(2,0,1).contiguous()
        table_q = table_q.permute(3,0,1,2).contiguous()
        table_k = table_k.permute(3,0,1,2).contiguous() #[L, h, hdim, 3]
        rel_idx = rel_idx.permute(1,0).contiguous()
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v9: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 50
        assert 512 % hdim == 0

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()

        table_q = table_q.permute(0,3,1,2).contiguous()
        table_k = table_k.permute(0,3,1,2).contiguous()  #[L, h, hdim, 3] -> [L, 3, h, hdim]
        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v7(N, M, h, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v7: backward time: {}s, M: {}".format(time.time() - t, M))
        
        grad_table_q = grad_table_q.permute(0,2,3,1).contiguous()
        grad_table_k = grad_table_k.permute(0,2,3,1).contiguous()

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, None, grad_k, None, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v9 = DotProdWithIdx_v9.apply


class DotProdWithIdxAll_v9(Function):
    @staticmethod
    def forward(ctx, q, index_q, index_q_offsets, n_max, k, index_k_offsets, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 50

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()
        
        q = q.permute(1,2,0).contiguous() #[h, hdim, L]
        k = k.permute(1,2,0).contiguous() #[h, hdim, L]
        table_q = table_q.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        table_k = table_k.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        rel_idx = rel_idx.permute(1,0).contiguous() #[3, M]

        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        # pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_all_forward_cuda_v9(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        
        q = q.permute(2,0,1).contiguous()
        k = k.permute(2,0,1).contiguous()
        table_q = table_q.permute(3,0,1,2).contiguous()
        table_k = table_k.permute(3,0,1,2).contiguous() #[L, h, hdim, 3]
        rel_idx = rel_idx.permute(1,0).contiguous()
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v9: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 50
        assert 512 % hdim == 0

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()

        table_q = table_q.permute(0,3,1,2).contiguous()
        table_k = table_k.permute(0,3,1,2).contiguous()  #[L, h, hdim, 3] -> [L, 3, h, hdim]
        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v7(N, M, h, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v7: backward time: {}s, M: {}".format(time.time() - t, M))
        
        grad_table_q = grad_table_q.permute(0,2,3,1).contiguous()
        grad_table_k = grad_table_k.permute(0,2,3,1).contiguous()


        grad_q_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_k_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()

        pointops_cuda.attention_step1_backward_cuda_v7(N, M, h, h*hdim, n_max, grad_output, index_q, index_q_offsets, index_k, index_k_offsets, q, k, grad_q_2, grad_k_2)
        
        grad_q += grad_q_2
        grad_k += grad_k_2

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, None, grad_k, None, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_all_v9 = DotProdWithIdxAll_v9.apply

class DotProdWithIdxAll_v10(Function):
    @staticmethod
    def forward(ctx, q, index_q, index_q_offsets, n_max, k, index_k_offsets, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 50

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # output = torch.cuda.FloatTensor(M, h).zero_()
        output = torch.cuda.FloatTensor(h, M).zero_()
        
        q = q.permute(1,2,0).contiguous() #[h, hdim, L]
        k = k.permute(1,2,0).contiguous() #[h, hdim, L]
        table_q = table_q.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        table_k = table_k.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        rel_idx = rel_idx.permute(1,0).contiguous() #[3, M]

        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        # pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_all_forward_cuda_v10(N, M, h, hdim, n_max, L, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        
        q = q.permute(2,0,1).contiguous()
        k = k.permute(2,0,1).contiguous()
        table_q = table_q.permute(3,0,1,2).contiguous()
        table_k = table_k.permute(3,0,1,2).contiguous() #[L, h, hdim, 3]
        rel_idx = rel_idx.permute(1,0).contiguous()
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v10: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 50
        assert 512 % hdim == 0

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()

        table_q = table_q.permute(0,3,1,2).contiguous()
        table_k = table_k.permute(0,3,1,2).contiguous()  #[L, h, hdim, 3] -> [L, 3, h, hdim]
        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v7(N, M, h, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_v7: backward time: {}s, M: {}".format(time.time() - t, M))
        
        grad_table_q = grad_table_q.permute(0,2,3,1).contiguous()
        grad_table_k = grad_table_k.permute(0,2,3,1).contiguous()


        grad_q_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_k_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()

        pointops_cuda.attention_step1_backward_cuda_v7(N, M, h, h*hdim, n_max, grad_output, index_q, index_q_offsets, index_k, index_k_offsets, q, k, grad_q_2, grad_k_2)
        
        grad_q += grad_q_2
        grad_k += grad_k_2

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, None, grad_k, None, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_all_v10 = DotProdWithIdxAll_v10.apply


class DotProdWithIdx_v5(Function):
    @staticmethod
    def forward(ctx, q, index_q_offsets, n_max, k, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        output = torch.cuda.FloatTensor(M, h).zero_()
            
        if n_max <= 1024:

            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()

            # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
            pointops_cuda.dot_prod_with_idx_forward_cuda_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("dot_prod_with_idx_v3: forward time: {}s, M: {}".format(time.time() - t, M))
        else:

            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()

            # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
            pointops_cuda.dot_prod_with_idx_forward_cuda_v6(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("dot_prod_with_idx_v6: forward time: {}s, M: {}".format(time.time() - t, M))

        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q_offsets, k, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, k, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]

        assert L <= 48
        assert hdim == 16
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        if n_max <= 1024:
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
                
            pointops_cuda.dot_prod_with_idx_backward_cuda_v5(N, M, h, hdim, L, n_max, grad_output, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("dot_prod_with_idx_v5: backward time: {}s, M: {}".format(time.time() - t, M))
        else:

            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
                
            pointops_cuda.dot_prod_with_idx_backward_cuda_v6(N, M, h, hdim, n_max, grad_output, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("dot_prod_with_idx_v6: backward time: {}s, M: {}".format(time.time() - t, M))

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}, M: {}".format(end - start, M))
        # input()
        return grad_q, None, None, grad_k, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v5 = DotProdWithIdx_v5.apply


class DotProdWithIdx_v4(Function):
    @staticmethod
    def forward(ctx, q, index_q, index_q_offsets, n_max, k, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        _, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        N = index_q_offsets.shape[0] - 1
        assert table_k.shape[0] == L

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        output = torch.cuda.FloatTensor(M, h).zero_()
        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v4(N, M, h, hdim, n_max, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output)
        
        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N_q, _, hdim = q.shape
        N_k = k.shape[0]
        L = table_q.shape[0]
        n_max = ctx.n_max
        N = index_q_offsets.shape[0] - 1
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.dot_prod_with_idx_backward_cuda_v4(N, M, h, hdim, n_max, grad_output, q, index_q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v9: {}".format(end - start))
        # input()
        return grad_q, None, None, None, grad_k, None, grad_table_q, grad_table_k, None

dot_prod_with_idx_v4 = DotProdWithIdx_v4.apply


class AttentionStep2WithRelPosValue(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        N_v, h, hdim = v.shape
        N_q = index0.max().item() + 1

        output = torch.cuda.FloatTensor(N_q, h, hdim).zero_()
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda(N_q, M, h, hdim, attn, v, index0, index1, table, rel_idx, output)

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.save_for_backward(attn, v, index0, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        attn, v, index0, index1, table, rel_idx = ctx.saved_tensors
        N_q, h, hdim = grad_output.shape
        N_v = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_v, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda(N_q, M, h, hdim, grad_output, index0, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}".format(end - start))
        # # input()
        return grad_attn, grad_v, None, None, grad_table, None

attention_step2_with_rel_pos_value = AttentionStep2WithRelPosValue.apply


class AttentionStep2WithRelPosValue_v2(Function):
    @staticmethod
    def forward(ctx, attn, v, index0_offsets, n_max, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v2(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)

        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v2: forward time: {}s, M: {}".format(time.time() - t, M))
        
        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0_offsets, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0_offsets, index1, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v2(N, M, h, hdim, n_max, grad_output, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v2: backward time: {}s, M: {}".format(time.time() - t, M))
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v2 = AttentionStep2WithRelPosValue_v2.apply

class AttentionStep2WithRelPosValue_v4(Function):
    @staticmethod
    def forward(ctx, attn, v, index0_offsets, n_max, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        L = table.shape[0]

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        assert L <= 60

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        if n_max <= 1024:
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
                
            pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step2_with_rel_pos_value_v4: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))
        else:
            
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
                
            pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v6(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step2_with_rel_pos_value_v6: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))
                
        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0_offsets, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0_offsets, index1, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        if n_max <= 1024:
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
            
            pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v4(N, M, h, hdim, L, n_max, grad_output, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step2_with_rel_pos_value_v4: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))
        else:
            
            if DEBUG:
                torch.cuda.synchronize()
                t = time.time()
            
            pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v6(N, M, h, hdim, n_max, grad_output, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
            
            if DEBUG:
                torch.cuda.synchronize()
                print("attn_step2_with_rel_pos_value_v6: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))
            
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v4 = AttentionStep2WithRelPosValue_v4.apply

class AttentionStep2WithRelPosValue_v7(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        L = table.shape[0]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        assert L <= 50

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        table = table.permute(0,3,1,2).contiguous() #[L, 3, h, hdim]

        # pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v7(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        # print("attn[:5,:5]: ", attn[:5, :5])
        table = table.permute(0,2,3,1).contiguous() #[L, h, hdim, 3]

        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v7: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        # grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_table = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()


        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        table = table.permute(1,2,3,0).contiguous()
        v = v.permute(1,2,0).contiguous()
        rel_idx = rel_idx.permute(1,0).contiguous()

        # print("grad_output.shape: {}, index0_offsets.shape: {}, index1.shape: {}, index1_offsets.shape: {}, attn.shape: {}, v.shape: {}, table.shape: {}, rel_idx.shape: {}, grad_attn.shape: {}, grad_v,.shape: {} grad_table.shape: {}".format(
        #     grad_output.shape, index0_offsets.shape, index1.shape, index1_offsets.shape, attn.shape, v.shape, table.shape, rel_idx.shape, grad_attn.shape, grad_v.shape, grad_table.shape
        # ))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v7(N, M, h, hdim, L, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v7: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        grad_table = grad_table.permute(0,2,3,1).contiguous()

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v7 = AttentionStep2WithRelPosValue_v7.apply

class AttentionStep2WithRelPosValue_v7p1(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        L = table.shape[0]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        assert L <= 50

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        table = table.permute(0,3,1,2).contiguous() #[L, 3, h, hdim]

        # pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v8(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        # print("attn[:5,:5]: ", attn[:5, :5])
        table = table.permute(0,2,3,1).contiguous() #[L, h, hdim, 3]

        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v8: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        # grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_table = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()


        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        table = table.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        v = v.permute(1,2,0).contiguous()
        rel_idx = rel_idx.permute(1,0).contiguous()

        # print("grad_output.shape: {}, index0_offsets.shape: {}, index1.shape: {}, index1_offsets.shape: {}, attn.shape: {}, v.shape: {}, table.shape: {}, rel_idx.shape: {}, grad_attn.shape: {}, grad_v,.shape: {} grad_table.shape: {}".format(
        #     grad_output.shape, index0_offsets.shape, index1.shape, index1_offsets.shape, attn.shape, v.shape, table.shape, rel_idx.shape, grad_attn.shape, grad_v.shape, grad_table.shape
        # ))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v7(N, M, h, hdim, L, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v7: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        grad_table = grad_table.permute(0,2,3,1).contiguous() #[L, h, hdim, 3]

        # print("grad_table.shape: ", grad_table.shape)

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v7p1 = AttentionStep2WithRelPosValue_v7p1.apply

class AttentionStep2WithRelPosValue_v10(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, table, rel_idx, n_interval=256):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        L = table.shape[0]

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        assert L <= 50

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        table = table.permute(0,3,1,2).contiguous() #[L, 3, h, hdim]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        # pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v7(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v7: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # print("attn[:5,:5]: ", attn[:5, :5])
        table = table.permute(0,2,3,1).contiguous() #[L, h, hdim, 3]

        ctx.n_max = n_max
        ctx.n_interval = n_interval
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        n_interval = ctx.n_interval
        attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(h, M).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        # grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        # grad_table = torch.cuda.FloatTensor(3, L, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(h, hdim, 3, L).zero_()


        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        table_transpose = table.permute(3,0,1,2) #[3, L, h, hdim]
        table = table.permute(1,2,3,0).contiguous() #[h, hdim, 3, L]
        v = v.permute(1,2,0).contiguous() #[h, hdim, N]
        rel_idx = rel_idx.permute(1,0).contiguous() #[3, M]
        grad_out_transpose = grad_output.permute(1,2,0).contiguous() #[h, hdim, N]
        attn_transpose = attn.permute(1,0).contiguous() #[h, M]

        # print("rel_idx_sort: ", rel_idx_sort)
        # print("sort_idx: ", sort_idx)

        # print("grad_output.shape: {}, index0_offsets.shape: {}, index1.shape: {}, index1_offsets.shape: {}, attn.shape: {}, v.shape: {}, table.shape: {}, rel_idx.shape: {}, grad_attn.shape: {}, grad_v,.shape: {} grad_table.shape: {}".format(
        #     grad_output.shape, index0_offsets.shape, index1.shape, index1_offsets.shape, attn.shape, v.shape, table.shape, rel_idx.shape, grad_attn.shape, grad_v.shape, grad_table.shape
        # ))

        # rel_idx_sort, sort_idx = rel_idx.sort(-1) #[3, M]
        # rel_idx_sort = rel_idx_sort.int()
        # sort_idx = sort_idx.int()
        # print("rel_idx_sort[0].bincount(): ", rel_idx_sort[0].bincount())
        # print("rel_idx_sort[1].bincount(): ", rel_idx_sort[1].bincount())
        # print("rel_idx_sort[2].bincount(): ", rel_idx_sort[2].bincount())
        rel_idx_sort = rel_idx
        sort_idx = rel_idx

        # with Timer(message='attn_step2_with_rel_pos_value_v10: sort time: '):
        #     rel_idx_sort, sort_idx = rel_idx.sort(-1) #[3, M]
        #     rel_idx_sort = rel_idx_sort.int()
        #     sort_idx = sort_idx.int()
            
        # with Timer(message='attn_step2_with_rel_pos_value_v10: argsort time: '):
        #     sort_idx_ = rel_idx.argsort(-1) #[3, M]
        #     # rel_idx_sort = rel_idx_sort.int()
        #     # sort_idx = sort_idx.int()
            
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        # print("attn_transpose.shape: {}, rel_idx.shape: {}, grad_out_transpose.shape: {}, grad_table.shape: {}".format(
        #     attn_transpose.shape, rel_idx.shape, grad_out_transpose.shape, grad_table.shape
        # ))

        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v10(N, M, h, hdim, L, n_max, n_interval, grad_output, grad_out_transpose, index0, index0_offsets, index1, index1_offsets, attn, attn_transpose, v, table, table_transpose, rel_idx, rel_idx_sort, sort_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v10: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        grad_attn = grad_attn.permute(1,0).contiguous()
        # grad_table = grad_table.permute(0,2,3,1).contiguous()
        # grad_table = grad_table.permute(1,2,3,0).contiguous() #[L, h, hdim, 3]
        grad_table = grad_table.permute(3,0,1,2).contiguous() #[h, hdim, 3, L] -> [L, h, hdim, 3]

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, None, grad_table, None, None

attention_step2_with_rel_pos_value_v10 = AttentionStep2WithRelPosValue_v10.apply

class AttentionStep2WithRelPosValue_v8(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, 3, h, hdim), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        L = table.shape[0]

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        assert L <= 50

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        # table = table.permute(0,3,1,2).contiguous() #[L, 3, h, hdim]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        # pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v7(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v8: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # print("attn[:5,:5]: ", attn[:5, :5])
        # table = table.permute(0,2,3,1).contiguous()

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        # grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_table = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()


        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        # table = table.permute(1,2,3,0).contiguous()
        table = table.permute(2,3,1,0).contiguous() #[L, 3, h, hdim] -> [h, hdim, 3, L]
        v = v.permute(1,2,0).contiguous()
        rel_idx = rel_idx.permute(1,0).contiguous()

        # print("grad_output.shape: {}, index0_offsets.shape: {}, index1.shape: {}, index1_offsets.shape: {}, attn.shape: {}, v.shape: {}, table.shape: {}, rel_idx.shape: {}, grad_attn.shape: {}, grad_v,.shape: {} grad_table.shape: {}".format(
        #     grad_output.shape, index0_offsets.shape, index1.shape, index1_offsets.shape, attn.shape, v.shape, table.shape, rel_idx.shape, grad_attn.shape, grad_v.shape, grad_table.shape
        # ))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        # print("N: {}, M: {}, h: {}, hdim: {}, L: {}, n_max: {}".format(N, M, h, hdim, L, n_max))

        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v7(N, M, h, hdim, L, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v8: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # grad_table = grad_table.permute(0,2,3,1).contiguous()

        del table, v, rel_idx

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v8 = AttentionStep2WithRelPosValue_v8.apply


class AttentionStep2WithRelPosValue_v9(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, pos_emb_v):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), pos_emb_v: (M, h, hdim)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and index1_offsets.is_contiguous() and pos_emb_v.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1
        # L = table.shape[0]

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        # assert L <= 50

        # if h >= 4 and h <= 8:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        # else:
        #     pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        # table = table.permute(0,3,1,2).contiguous() #[L, 3, h, hdim]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        # print("N: {}, M: {}, h: {}, hdim: {}, n_max: {}".format(N, M, h, hdim, n_max))

        # pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v4(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v9(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, pos_emb_v, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v9: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # print("attn[:5,:5]: ", attn[:5, :5])
        # table = table.permute(0,2,3,1).contiguous()

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, pos_emb_v)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets, pos_emb_v = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        # L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and pos_emb_v.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_pos_emb_v = torch.cuda.FloatTensor(M, h, hdim).zero_()
        # grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        # grad_table = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()


        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        # table = table.permute(1,2,3,0).contiguous()
        v = v.permute(1,2,0).contiguous()
        pos_emb_v = pos_emb_v.permute(1,2,0).contiguous()
        # rel_idx = rel_idx.permute(1,0).contiguous()

        # print("grad_output.shape: {}, index0_offsets.shape: {}, index1.shape: {}, index1_offsets.shape: {}, attn.shape: {}, v.shape: {}, table.shape: {}, rel_idx.shape: {}, grad_attn.shape: {}, grad_v,.shape: {} grad_table.shape: {}".format(
        #     grad_output.shape, index0_offsets.shape, index1.shape, index1_offsets.shape, attn.shape, v.shape, table.shape, rel_idx.shape, grad_attn.shape, grad_v.shape, grad_table.shape
        # ))

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v9(N, M, h, hdim, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, pos_emb_v, grad_attn, grad_v, grad_pos_emb_v)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v9: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        # grad_table = grad_table.permute(0,2,3,1).contiguous()

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, None, grad_pos_emb_v

attention_step2_with_rel_pos_value_v9 = AttentionStep2WithRelPosValue_v9.apply


class AttentionStep2WithRelPosValue_v5(Function):
    @staticmethod
    def forward(ctx, attn, v, index0_offsets, n_max, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        # N_q = int(index0_offsets.max().item()) + 1

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v5(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)

        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v5: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))
        
        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0_offsets, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0_offsets, index1, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        # N = v.shape[0]
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v4(N, M, h, hdim, n_max, grad_output, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value_v4: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v11: {}, M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v5 = AttentionStep2WithRelPosValue_v5.apply

class AttentionStep2WithRelPosValue_v3(Function):
    @staticmethod
    def forward(ctx, attn, v, index0_offsets, n_max, index0, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        N_v, h, hdim = v.shape
        # N_q = int(index0_offsets.max().item()) + 1
        N = index0_offsets.shape[0] - 1

        output = torch.cuda.FloatTensor(N_v, h, hdim).zero_()
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v3(N, M, h, hdim, n_max, attn, v, index0_offsets, index0, index1, table, rel_idx, output)

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0_offsets, index0, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0_offsets, index0, index1, table, rel_idx = ctx.saved_tensors
        _, h, hdim = grad_output.shape
        N_v = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]
        N = index0_offsets.shape[0] - 1

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())

        # print("attn.is_contiguous(): {}, v.is_contiguous(): {}, index0_offsets.is_contiguous(): {}, index1.is_contiguous(): {}, grad_output.is_contiguous(): {}, table.is_contiguous(): {}, rel_idx.is_contiguous(): {}".format(
        #     attn.is_contiguous(), v.is_contiguous(), index0_offsets.is_contiguous(), index1.is_contiguous(), grad_output.is_contiguous(), table.is_contiguous(), rel_idx.is_contiguous()
        # ))

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index0.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_v, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()
        
        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v3(N, M, h, hdim, n_max, grad_output, index0_offsets, index0, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v12: {} M: {}".format(end - start, M))
        
        return grad_attn, grad_v, None, None, None, None, grad_table, None

attention_step2_with_rel_pos_value_v3 = AttentionStep2WithRelPosValue_v3.apply

def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True, return_indx=False):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    # 相对位置
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)
    if use_xyz:
        if return_indx:
            return torch.cat((grouped_xyz, grouped_feat), -1), idx # (m, nsample, 3+c)
        else:
            return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        if return_indx:
            return grouped_feat, idx
        else:
            return grouped_feat


def Divide2Patch(nsample, xyz, offset, return_offset=False, anchor_scale=None):
    # nsample: 16  xyz: (n, 3)  offset: (b)
    downsample_scale = anchor_scale or nsample
    new_offset, count = [offset[0].item() // downsample_scale], offset[0].item() // downsample_scale
    for i in range(1, offset.shape[0]):
        count += (offset[i].item() - offset[i-1].item()) // downsample_scale
        new_offset.append(count)
    # print("donw sample scale:", downsample_scale,"offset:", offset, "newoffset:", new_offset)
    new_offset = torch.cuda.IntTensor(new_offset)
    idx = furthestsampling(xyz, offset, new_offset) # (m)
    new_xyz = xyz[idx.long()]
    p_idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)
    if return_offset:
        return p_idx, new_offset
    else:
        return p_idx
    
class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape; nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        idx, = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(n, nsample, c, idx, grad_output, grad_input1, grad_input2)
        return grad_input1, grad_input2, None

subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert input.is_contiguous() and position.is_contiguous() and weight.is_contiguous()
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(n, nsample, c, w_c, input, position, weight, idx, output)
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight)
        return grad_input, grad_position, grad_weight, None

aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


def interpolation_v2(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()

    idx, _ = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)

    # print("e3: idx.shape: {}, idx[:5]: {}".format(idx.shape, idx[:5]))

    dist = torch.sqrt(((new_xyz.unsqueeze(1) - xyz[idx.long()]) ** 2).sum(-1) + 1e-8)

    # print("e4: dist.shape: {}, dist[:5]: {}".format(dist.shape, dist[:5]))
    # print("((_-dist)**2).max(): ", ((_-dist)**2).max())
    # input()

    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8) # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        return None, None, grad_input, None, None, None

interpolation2 = Interpolation.apply
