import pytest
from pytest import mark as m
from hypothesis import given, strategies as st
import torch

from sparse import (
    strided_transpose,
    get_full_attention_mask,
    get_mask_with_bandwidth,
    get_strided_mask,
    fixed_attention_mask,
)


@m.describe("fn strided_transpose")
class Test_strided_transpose:
    @m.it("returns a tensor with shape (n, t, embd)")
    def test_strided_transpose_shape(self):
        n, t, embd = 2, 16, 4
        n_ctx = 16
        local_attn_ctx = 4
        blocksize = 2
        x = torch.randn(n, t, embd)
        y = strided_transpose(x, local_attn_ctx)
        assert y.shape == (n, t, embd)

    def test_strided_transpose_different_shapes(self):
        # Test with various input shapes
        shapes = [(2, 16, 4), (4, 32, 8), (1, 8, 2)]
        n_ctx = 16
        local_attn_ctx = 4
        blocksize = 2
        for shape in shapes:
            x = torch.randn(shape)
            y = strided_transpose(x, local_attn_ctx)
            assert y.shape == shape

    def test_strided_transpose(self):
        shape = (2, 8, 4)
        size = shape[0] * shape[1] * shape[2]
        x = torch.arange(size).view(shape)
        y = strided_transpose(x, 1)
        assert(torch.equal(y, x))

        y = strided_transpose(x, 2)
        expected = torch.tensor([[
            [ 0,  1,  2,  3],
            [ 8,  9, 10, 11],
            [16, 17, 18, 19],
            [24, 25, 26, 27],
            [ 4,  5,  6,  7],
            [12, 13, 14, 15],
            [20, 21, 22, 23],
            [28, 29, 30, 31]],
           [[32, 33, 34, 35],
            [40, 41, 42, 43],
            [48, 49, 50, 51],
            [56, 57, 58, 59],
            [36, 37, 38, 39],
            [44, 45, 46, 47],
            [52, 53, 54, 55],
            [60, 61, 62, 63]]])
        assert torch.equal(y, expected)
        y = strided_transpose(x, 4)
        expected = torch.tensor([
            [[ 0,  1,  2,  3],
            [16, 17, 18, 19],
            [ 4,  5,  6,  7],
            [20, 21, 22, 23],
            [ 8,  9, 10, 11],
            [24, 25, 26, 27],
            [12, 13, 14, 15],
            [28, 29, 30, 31]],

            [[32, 33, 34, 35],
            [48, 49, 50, 51],
            [36, 37, 38, 39],
            [52, 53, 54, 55],
            [40, 41, 42, 43],
            [56, 57, 58, 59],
            [44, 45, 46, 47],
            [60, 61, 62, 63]]])
        assert torch.equal(y, expected)

@m.describe("fn get_full_attention_mask")
@m.it('works')
def test_full_attention_mask():
    mask = get_full_attention_mask(4)
    expected = torch.tensor([
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
    assert mask.shape == expected.shape
    assert torch.allclose(mask, expected)

@m.describe("fn get_mask_with_bandwidth")
class Test_get_mask_with_bandwidth:
    @m.context('bandwidth<1')
    @m.it('contracts the mask')
    def test_local_contraction(self):
        mask = get_mask_with_bandwidth(4, 0)
        expected = torch.tensor([
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)
        mask = get_mask_with_bandwidth(4, -1)
        expected = torch.tensor([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 1., 0., 0.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)

    @m.context('bandwidth=1')
    @m.it('is the same as "all"')
    def test_bandwidth_1(self):
        mask = get_mask_with_bandwidth(10, 1)
        expected = get_full_attention_mask(10)
        assert torch.allclose(mask, expected)

    @m.context('bandwidth>1')
    def test_local_expansion(self):
        mask = get_mask_with_bandwidth(4, 2)
        expected = torch.tensor([
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)
        mask = get_mask_with_bandwidth(4, 3)
        expected = torch.tensor([
            [1., 1., 1., 0.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)

@m.describe("fn get_strided_mask")
class Test_get_strided_mask:
    @m.context('local_attn_ctx=1')
    @m.it('is the same as "all"')
    def test_strided_1(self):
        mask = get_strided_mask(10, 1)
        expected = get_full_attention_mask(10)
        assert torch.allclose(mask, expected)

    @m.it('uses local_attn_ctx as stride spacing')
    def test_strided(self):
        mask = get_strided_mask(6, 2)
        expected = torch.tensor([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [1., 0., 1., 0., 0., 0.],
            [0., 1., 0., 1., 0., 0.],
            [1., 0., 1., 0., 1., 0.],
            [0., 1., 0., 1., 0., 1.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)
        mask = get_strided_mask(6, 3)
        expected = torch.tensor([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [1., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0., 1.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)

    @m.it('stride less than 1 is an error')
    def test_stride_too_small(self):
        with pytest.raises(ValueError):
            get_strided_mask(6, 0)

@m.describe('fn fixed_attention_mask')
class Test_fixed_attention_mask:
    @m.it('do')
    def test_something(self):
        mask = fixed_attention_mask(seq_len=10, stride=4)
        expected = torch.tensor([
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 1., 1., 0., 0.],
            [0., 1., 0., 0., 0., 1., 1., 1., 1., 0.],
            [0., 1., 0., 0., 0., 1., 1., 1., 1., 1.]])
        assert mask.shape == expected.shape
        assert torch.allclose(mask, expected)

