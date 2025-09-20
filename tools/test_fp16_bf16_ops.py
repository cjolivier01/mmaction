import os
import sys
import copy
import torch


def main():
    assert torch.cuda.is_available(), 'CUDA required for these tests.'
    # Ensure extension dirs are importable when running from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, 'mmaction/ops/trajectory_conv_package'))
    sys.path.insert(0, os.path.join(repo_root, 'mmaction/ops/roi_align'))
    sys.path.insert(0, os.path.join(repo_root, 'mmaction/ops/roi_pool'))

    cc = torch.cuda.get_device_capability()
    has_bf16 = (cc[0] >= 8)

    def test_roi_align(dtype, check_parity=False):
        from mmaction.ops.roi_align.modules.roi_align import RoIAlign
        torch.cuda.manual_seed(0)
        feats32 = torch.randn(2, 4, 8, 8, device='cuda', dtype=torch.float32, requires_grad=True)
        rois32 = torch.tensor([[0, 1.0, 1.0, 5.0, 6.0], [1, 0.5, 0.5, 7.5, 7.5]], device='cuda', dtype=torch.float32)
        op = RoIAlign(out_size=(3, 3), spatial_scale=1.0, sample_num=2)
        out32 = op(feats32, rois32)
        loss32 = out32.sum(); loss32.backward()
        feats = feats32.detach().to(dtype).requires_grad_(True)
        rois = rois32.to(dtype)
        out = op(feats, rois)
        loss = out.sum(); loss.backward()
        assert torch.isfinite(loss).item(), 'roi_align loss not finite'
        if check_parity:
            rtol, atol = ((5e-2, 1e-2) if dtype == torch.float16 else (8e-2, 2e-2))
            torch.testing.assert_close(out.detach().to(torch.float32), out32.detach(), rtol=rtol, atol=atol)
            torch.testing.assert_close(feats.grad.detach().to(torch.float32), feats32.grad.detach(), rtol=rtol, atol=atol)

    def test_roi_pool(dtype, check_parity=False):
        from mmaction.ops.roi_pool.modules.roi_pool import RoIPool
        torch.cuda.manual_seed(0)
        feats32 = torch.randn(2, 4, 8, 8, device='cuda', dtype=torch.float32, requires_grad=True)
        rois32 = torch.tensor([[0, 1.0, 1.0, 5.0, 6.0], [1, 0.5, 0.5, 7.5, 7.5]], device='cuda', dtype=torch.float32)
        op = RoIPool(out_size=(3, 3), spatial_scale=1.0)
        out32 = op(feats32, rois32)
        loss32 = out32.sum(); loss32.backward()
        feats = feats32.detach().to(dtype).requires_grad_(True)
        rois = rois32.to(dtype)
        out = op(feats, rois)
        loss = out.sum(); loss.backward()
        assert torch.isfinite(loss).item(), 'roi_pool loss not finite'
        if check_parity:
            rtol, atol = ((5e-2, 1e-2) if dtype == torch.float16 else (8e-2, 2e-2))
            torch.testing.assert_close(out.detach().to(torch.float32), out32.detach(), rtol=rtol, atol=atol)
            torch.testing.assert_close(feats.grad.detach().to(torch.float32), feats32.grad.detach(), rtol=rtol, atol=atol)

    def test_traj_conv(dtype, check_parity=False):
        from mmaction.ops.trajectory_conv_package.traj_conv import TrajConv
        torch.cuda.manual_seed(0)
        N, C, T, H, W = 2, 4, 3, 6, 6
        x32 = torch.randn(N, C, T, H, W, device='cuda', dtype=torch.float32, requires_grad=True)
        module32 = TrajConv(in_channels=C, out_channels=5, kernel_size=(1, 3, 3), stride=1,
                            padding=(0, 1, 1), dilation=1, num_deformable_groups=1,
                            im2col_step=2, bias=True).cuda().to(torch.float32)
        with torch.no_grad():
            pad = (0, 1, 1); dil = (1, 1, 1); ks = (1, 3, 3); stride = (1, 1, 1)
            T_out = (T + 2 * pad[0] - (dil[0] * (ks[0] - 1) + 1)) // stride[0] + 1
            H_out = (H + 2 * pad[1] - (dil[1] * (ks[1] - 1) + 1)) // stride[1] + 1
            W_out = (W + 2 * pad[2] - (dil[2] * (ks[2] - 1) + 1)) // stride[2] + 1
        offC = 2 * ks[0] * ks[1] * ks[2] * 1
        offset32 = torch.zeros((N, offC, T_out, H_out, W_out), device='cuda', dtype=torch.float32, requires_grad=True)
        # baseline
        y32 = module32(x32, offset32)
        loss32 = y32.sum(); loss32.backward()
        # dtype run
        module = copy.deepcopy(module32).to(dtype)
        x = x32.detach().to(dtype).requires_grad_(True)
        offset = offset32.detach().to(dtype).requires_grad_(True)
        y = module(x, offset)
        loss = y.sum(); loss.backward()
        assert torch.isfinite(loss).item(), 'traj_conv loss not finite'
        if check_parity:
            rtol, atol = ((5e-2, 1e-2) if dtype == torch.float16 else (8e-2, 2e-2))
            torch.testing.assert_close(y.detach().to(torch.float32), y32.detach(), rtol=rtol, atol=atol)
            torch.testing.assert_close(x.grad.detach().to(torch.float32), x32.grad.detach(), rtol=rtol, atol=atol)
            torch.testing.assert_close(offset.grad.detach().to(torch.float32), offset32.grad.detach(), rtol=rtol, atol=atol)

    dtypes = [torch.float16]
    if has_bf16:
        dtypes.append(torch.bfloat16)

    for dt in dtypes:
        # smoke
        test_roi_align(dt)
        test_roi_pool(dt)
        test_traj_conv(dt)
        # parity vs fp32
        test_roi_align(dt, check_parity=True)
        test_roi_pool(dt, check_parity=True)
        test_traj_conv(dt, check_parity=True)

    print('fp16/bf16 smoke + parity tests passed (where supported).')


if __name__ == '__main__':
    main()
