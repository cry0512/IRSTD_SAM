import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class PhasePromptGenerator(nn.Module):
    """Phase-based saliency prompt generator (PFT)."""

    def __init__(
        self,
        top_k: int = 5,
        input_size=(256, 256),
        min_dist: int = 10,
        saliency_thr: float = 0.1,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
        use_window: bool = True,
        border_width: int = 12,
        dynamic_thr: bool = True,
        dynamic_thr_quantile: float = 0.9,
        dynamic_thr_mode: str = "max",
        dynamic_top_k: bool = True,
        min_top_k: int = 1,
        use_dct: bool = False,
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.input_size = input_size
        self.min_dist = int(min_dist)
        self.saliency_thr = float(saliency_thr)
        self.blur_kernel_size = int(blur_kernel_size)
        self.sigma = float(blur_sigma)
        self.use_window = bool(use_window)
        self.border_width = int(border_width)
        self.dynamic_thr = bool(dynamic_thr)
        self.dynamic_thr_quantile = float(dynamic_thr_quantile)
        self.dynamic_thr_mode = str(dynamic_thr_mode)
        self.dynamic_top_k = bool(dynamic_top_k)
        self.min_top_k = int(min_top_k)
        self.use_dct = bool(use_dct)
        self._window_cache = {}
        self._border_cache = {}

    def _get_window(self, h: int, w: int, device, dtype):
        key = (h, w, device, dtype)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached
        wy = torch.hann_window(h, device=device, dtype=dtype, periodic=False)
        wx = torch.hann_window(w, device=device, dtype=dtype, periodic=False)
        window = (wy[:, None] * wx[None, :]).view(1, 1, h, w)
        self._window_cache[key] = window
        return window

    def _get_border_mask(self, h: int, w: int, device, dtype):
        bw = max(0, int(self.border_width))
        if bw <= 0:
            return None
        key = (h, w, bw, device, dtype)
        cached = self._border_cache.get(key)
        if cached is not None:
            return cached
        mask = torch.ones((1, 1, h, w), device=device, dtype=dtype)
        mask[:, :, :bw, :] = 0
        mask[:, :, -bw:, :] = 0
        mask[:, :, :, :bw] = 0
        mask[:, :, :, -bw:] = 0
        self._border_cache[key] = mask
        return mask

    def _dynamic_threshold(self, saliency_map: torch.Tensor) -> torch.Tensor:
        bsz = saliency_map.shape[0]
        flat = saliency_map.view(bsz, -1)
        n = flat.shape[1]
        q = min(max(self.dynamic_thr_quantile, 0.0), 1.0)
        k = max(1, int((1.0 - q) * n))
        vals, _ = torch.topk(flat, k, dim=1)
        thr = vals[:, -1].view(bsz, 1, 1, 1)
        return thr

    def _pad_points(self, coords_list, labels_list):
        max_k = 0
        device = None
        dtype = None
        for coords in coords_list:
            if coords is not None:
                max_k = max(max_k, coords.shape[0])
                if device is None:
                    device = coords.device
                    dtype = coords.dtype
        max_k = max(max_k, 1)
        if device is None:
            for labels in labels_list:
                if labels is not None:
                    device = labels.device
                    break
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        coords_padded = []
        labels_padded = []
        for coords, labels in zip(coords_list, labels_list):
            if coords is None:
                coords = torch.zeros((0, 2), device=device, dtype=dtype)
            if labels is None:
                labels = torch.zeros((0,), device=device, dtype=torch.int)
            if coords.shape[0] < max_k:
                pad_len = max_k - coords.shape[0]
                coords = F.pad(coords, (0, 0, 0, pad_len), value=-1.0)
                labels = F.pad(labels, (0, pad_len), value=-1.0)
            coords_padded.append(coords)
            labels_padded.append(labels)
        return torch.stack(coords_padded, 0), torch.stack(labels_padded, 0)

    def label_points_by_gt(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        gt_mask: torch.Tensor,
        saliency_map: torch.Tensor = None,
        min_pos: int = 1,
        max_neg: int = 2,
    ):
        bsz, k, _ = point_coords.shape
        coords_list = []
        labels_list = []
        for b in range(bsz):
            coords = point_coords[b]
            labels = point_labels[b]
            valid = labels >= 0
            coords = coords[valid]
            if coords.numel() == 0:
                coords = coords.reshape(0, 2)
            gt = gt_mask[b]
            h, w = gt.shape
            if coords.numel() > 0:
                xs = coords[:, 0].round().long().clamp(0, w - 1)
                ys = coords[:, 1].round().long().clamp(0, h - 1)
                inside = gt[ys, xs] > 0
            else:
                inside = torch.zeros((0,), device=gt.device, dtype=torch.bool)
                xs = ys = torch.zeros((0,), device=gt.device, dtype=torch.long)
            pos_coords = coords[inside]
            neg_coords = coords[~inside]

            if max_neg is not None and max_neg >= 0 and neg_coords.shape[0] > max_neg:
                if saliency_map is not None and neg_coords.shape[0] > 0:
                    sal = saliency_map[b, 0]
                    neg_sal = sal[ys[~inside], xs[~inside]]
                    idx = torch.topk(neg_sal, k=max_neg).indices
                    neg_coords = neg_coords[idx]
                else:
                    neg_coords = neg_coords[:max_neg]

            has_gt = bool(gt.max().item() > 0)
            if min_pos > 0 and has_gt and pos_coords.shape[0] < min_pos:
                pos_idx = (gt > 0).nonzero(as_tuple=False)
                if pos_idx.numel() > 0:
                    need = min_pos - pos_coords.shape[0]
                    sel = pos_idx[torch.randint(len(pos_idx), (need,), device=pos_idx.device)]
                    extra = torch.stack([sel[:, 1], sel[:, 0]], dim=1).to(coords.dtype)
                    pos_coords = torch.cat([pos_coords, extra], dim=0)

            if self.top_k is not None and self.top_k > 0:
                max_total = int(self.top_k)
                if pos_coords.shape[0] > max_total:
                    pos_coords = pos_coords[:max_total]
                    neg_coords = neg_coords[:0]
                else:
                    keep_neg = max_total - pos_coords.shape[0]
                    neg_coords = neg_coords[:keep_neg]

            coords_out = torch.cat([pos_coords, neg_coords], dim=0)
            labels_out = torch.cat(
                [
                    torch.ones(pos_coords.shape[0], device=coords_out.device, dtype=torch.int),
                    torch.zeros(neg_coords.shape[0], device=coords_out.device, dtype=torch.int),
                ],
                dim=0,
            )
            coords_list.append(coords_out)
            labels_list.append(labels_out)

        return self._pad_points(coords_list, labels_list)

    def select_negatives_from_mask(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        saliency_map: torch.Tensor,
        mask_bhw: torch.Tensor,
        max_neg: int,
    ):
        bsz, _, _ = point_coords.shape
        coords_list = []
        labels_list = []
        for b in range(bsz):
            coords = point_coords[b]
            labels = point_labels[b]
            valid = labels >= 0
            coords = coords[valid]
            if coords.numel() == 0 or max_neg <= 0:
                coords_list.append(torch.zeros((0, 2), device=point_coords.device, dtype=point_coords.dtype))
                labels_list.append(torch.zeros((0,), device=point_coords.device, dtype=torch.int))
                continue
            mask = mask_bhw[b]
            h, w = mask.shape
            xs = coords[:, 0].round().long().clamp(0, w - 1)
            ys = coords[:, 1].round().long().clamp(0, h - 1)
            outside = mask[ys, xs] <= 0
            neg_coords = coords[outside]
            if neg_coords.shape[0] > 0:
                sal = saliency_map[b, 0]
                neg_sal = sal[ys[outside], xs[outside]]
                k = min(max_neg, neg_coords.shape[0])
                idx = torch.topk(neg_sal, k=k).indices
                neg_coords = neg_coords[idx]
            else:
                neg_coords = neg_coords[:0]
            labels_out = torch.zeros(neg_coords.shape[0], device=point_coords.device, dtype=torch.int)
            coords_list.append(neg_coords)
            labels_list.append(labels_out)

        return self._pad_points(coords_list, labels_list)

    def _dct_1d(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x = x.movedim(dim, -1)
        n = x.shape[-1]
        x_ext = torch.cat([x, x.flip(-1)], dim=-1)
        X = torch.fft.rfft(x_ext, dim=-1)
        k = torch.arange(n, device=x.device, dtype=x.dtype)
        factor = torch.exp(-1j * math.pi * k / (2 * n))
        X = X[..., :n] * factor
        y = X.real * 2
        return y.movedim(-1, dim)

    def _idct_1d(self, X: torch.Tensor, dim: int) -> torch.Tensor:
        X = X.movedim(dim, -1)
        n = X.shape[-1]
        k = torch.arange(n, device=X.device, dtype=X.dtype)
        factor = torch.exp(1j * math.pi * k / (2 * n))
        V = torch.complex(X / 2, torch.zeros_like(X)) * factor
        V_full = torch.zeros(X.shape[:-1] + (n + 1,), device=X.device, dtype=V.dtype)
        V_full[..., :n] = V
        V_full[..., n] = 0
        y = torch.fft.irfft(V_full, n=2 * n, dim=-1)
        y = y[..., :n]
        return y.movedim(-1, dim)

    def _dct_2d(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dct_1d(x, dim=-1)
        x = self._dct_1d(x, dim=-2)
        return x

    def _idct_2d(self, x: torch.Tensor) -> torch.Tensor:
        x = self._idct_1d(x, dim=-1)
        x = self._idct_1d(x, dim=-2)
        return x

    def get_phase_saliency(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self.use_window:
            window = self._get_window(img_tensor.shape[-2], img_tensor.shape[-1], img_tensor.device, img_tensor.dtype)
            img_tensor = img_tensor * window
        if self.use_dct:
            coeff = self._dct_2d(img_tensor)
            phase = torch.sign(coeff)
            phase = torch.where(phase == 0, torch.ones_like(phase), phase)
            recon = self._idct_2d(phase)
        else:
            freq = torch.fft.fft2(img_tensor)
            phase_only_freq = torch.exp(1j * torch.angle(freq))
            recon = torch.fft.ifft2(phase_only_freq).real
        saliency = recon.pow(2)
        saliency = saliency.mean(dim=1, keepdim=True)
        saliency = self._gaussian_blur(saliency)
        if self.border_width > 0:
            mask = self._get_border_mask(saliency.shape[-2], saliency.shape[-1], saliency.device, saliency.dtype)
            if mask is not None:
                saliency = saliency * mask

        bsz = saliency.shape[0]
        flat = saliency.view(bsz, -1)
        minv = flat.min(dim=1, keepdim=True).values.view(bsz, 1, 1, 1)
        maxv = flat.max(dim=1, keepdim=True).values.view(bsz, 1, 1, 1)
        saliency = (saliency - minv) / (maxv - minv + 1e-6)
        return saliency

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        if self.blur_kernel_size <= 1 or self.sigma <= 0:
            return x
        k_size = self.blur_kernel_size
        if k_size % 2 == 0:
            k_size += 1
        return TF.gaussian_blur(x, kernel_size=[k_size, k_size], sigma=[self.sigma, self.sigma])

    def extract_points(self, saliency_map: torch.Tensor):
        bsz, _, h, w = saliency_map.shape
        device = saliency_map.device

        k_size = max(1, int(self.min_dist))
        if k_size % 2 == 0:
            k_size += 1
        padding = k_size // 2
        local_max = F.max_pool2d(saliency_map, kernel_size=k_size, stride=1, padding=padding)
        thr = torch.full((bsz, 1, 1, 1), float(self.saliency_thr), device=device, dtype=saliency_map.dtype)
        if self.dynamic_thr:
            dyn_thr = self._dynamic_threshold(saliency_map)
            if self.dynamic_thr_mode == "replace":
                thr = dyn_thr
            else:
                thr = torch.maximum(thr, dyn_thr)
        is_local_max = (saliency_map == local_max) & (saliency_map > thr)

        point_coords_batch = []
        point_labels_batch = []
        lengths = []
        for b in range(bsz):
            s_map = saliency_map[b, 0]
            mask = is_local_max[b, 0]
            y_idx, x_idx = torch.where(mask)

            if x_idx.numel() == 0:
                flat_idx = torch.argmax(s_map)
                fallback_y = flat_idx // w
                fallback_x = flat_idx % w
                points = torch.stack([fallback_x, fallback_y], dim=0).unsqueeze(0).float()
            else:
                vals = s_map[y_idx, x_idx]
                k = min(self.top_k, vals.numel())
                _, topk_indices = torch.topk(vals, k)
                points = torch.stack([x_idx[topk_indices], y_idx[topk_indices]], dim=1).float()

            if self.dynamic_top_k:
                if points.size(0) < self.min_top_k:
                    pad_len = self.min_top_k - points.size(0)
                    last_point = points[-1:].repeat(pad_len, 1)
                    points = torch.cat([points, last_point], dim=0)
                k_use = min(self.top_k, points.size(0))
                points = points[:k_use]
                labels = torch.ones(k_use, dtype=torch.int, device=device)
                lengths.append(k_use)
            else:
                if points.size(0) < self.top_k:
                    pad_len = self.top_k - points.size(0)
                    last_point = points[-1:].repeat(pad_len, 1)
                    points = torch.cat([points, last_point], dim=0)
                points = points[: self.top_k]
                labels = torch.ones(self.top_k, dtype=torch.int, device=device)

            point_coords_batch.append(points)
            point_labels_batch.append(labels)

        if self.dynamic_top_k:
            max_k = max(lengths) if lengths else max(1, self.min_top_k)
            coords_padded = []
            labels_padded = []
            for points, labels in zip(point_coords_batch, point_labels_batch):
                if points.size(0) < max_k:
                    pad_len = max_k - points.size(0)
                    points = F.pad(points, (0, 0, 0, pad_len), value=-1.0)
                    labels = F.pad(labels, (0, pad_len), value=-1.0)
                coords_padded.append(points)
                labels_padded.append(labels)
            point_coords = torch.stack(coords_padded, 0)
            point_labels = torch.stack(labels_padded, 0)
        else:
            point_coords = torch.stack(point_coords_batch, 0)
            point_labels = torch.stack(point_labels_batch, 0)
        return point_coords, point_labels

    def forward(self, images: torch.Tensor):
        saliency_map = self.get_phase_saliency(images)
        point_coords, point_labels = self.extract_points(saliency_map)
        return point_coords, point_labels, saliency_map


class HQSAM_PGAP_Wrapper(nn.Module):
    def __init__(self, hq_sam_model, top_k: int = 3, min_dist: int = 15):
        super().__init__()
        self.sam = hq_sam_model
        self.pgap = PhasePromptGenerator(top_k=top_k, min_dist=min_dist)

        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_images: torch.Tensor):
        with torch.no_grad():
            if hasattr(self.sam, "get_image_embeddings"):
                image_embeddings, interms = self.sam.get_image_embeddings(input_images)
            else:
                image_embeddings = self.sam.image_encoder(input_images)
                interms = None

            point_coords, point_labels, saliency_map = self.pgap(input_images)

            if point_coords.dim() == 3:
                point_coords = point_coords.unsqueeze(1)
            if point_labels.dim() == 2:
                point_labels = point_labels.unsqueeze(1)

            if hasattr(self.sam, "predict_masks"):
                h, w = input_images.shape[-2:]
                masks, _ = self.sam.predict_masks(
                    image_embeddings,
                    interms,
                    point_coords,
                    point_labels,
                    multimask_output=False,
                    input_h=h,
                    input_w=w,
                    output_h=h,
                    output_w=w,
                    hq_token_only=False,
                )
                masks = masks[:, 0, 0]
            else:
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                low_res_masks, _ = self.sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    hq_token_only=False,
                    interm_embeddings=interms,
                )
                h, w = input_images.shape[-2:]
                masks = F.interpolate(
                    low_res_masks,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

        return masks, saliency_map
