# mypy: allow-untyped-defs

from __future__ import annotations

import logging
import math
import time
import warnings
from typing import Any, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from espnet2.optimizers.named_optimizer import NamedOptimizer


class KFACStore:
    """Load and store damped inverse KFAC factors for CGaLore.

    Expected ``torch.save`` format::

        {
            "encoder.layers.0.self_attn.linear_q": {
                "Q": Tensor[d_in, d_in] or list[Tensor],
                "H": Tensor[d_out, d_out] or list[Tensor],
            },
            ...
        }

    Q = E[x x^T]         input covariance
    H = E[delta delta^T] output-gradient covariance

    For each layer we precompute

        Q_inv = (Q + damping I)^(-1)
        H_inv = (H + damping I)^(-1)

    and store the inverses on CPU. They are moved to the gradient device only
    when the GaLore basis is refreshed.
    """

    def __init__(
        self,
        path: str,
        damping: float = 1e-3,
        precompute_device: str = "auto",
        log_every: int = 1,
        allowed_layer_names: set[str] | None = None,
    ) -> None:
        if damping <= 0:
            raise ValueError(f"kfac_damping must be > 0, got {damping}.")
        if precompute_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "kfac_precompute_device must be one of auto/cpu/cuda, "
                f"got {precompute_device}."
            )

        self.path = path
        self.damping = float(damping)
        self.precompute_device = precompute_device
        self.log_every = int(log_every)

        device = self._resolve_precompute_device(precompute_device)
        raw_stats = torch.load(path, map_location="cpu")
        self.stats: dict[str, dict[str, Tensor]] = {}

        allowed_keys = None
        if allowed_layer_names is not None:
            allowed_keys = set()
            for name in allowed_layer_names:
                allowed_keys.add(name)
                allowed_keys.add(name + ".weight")
                allowed_keys.add(name.removesuffix(".weight"))

        logging.info(
            "KFACStore preprocessing: entries=%d, damping=%g, device=%s",
            len(raw_stats),
            self.damping,
            device,
        )

        t_total = time.perf_counter()
        for idx, (layer_name, entry) in enumerate(raw_stats.items(), 1):
            if allowed_keys is not None and layer_name not in allowed_keys:
                continue

            q = self._unwrap_factor(entry["Q"]).detach().to(
                device=device, dtype=torch.float32
            )
            h = self._unwrap_factor(entry["H"]).detach().to(
                device=device, dtype=torch.float32
            )

            q_inv = self._damped_inverse(q, self.damping).cpu().contiguous()
            h_inv = self._damped_inverse(h, self.damping).cpu().contiguous()

            self.stats[layer_name] = {
                "Q_inv": q_inv,
                "H_inv": h_inv,
            }

            if idx == 1 or idx % self.log_every == 0:
                logging.info(
                    "KFACStore processed %d/%d: %s, Q=%s, H=%s",
                    idx,
                    len(raw_stats),
                    layer_name,
                    tuple(q.shape),
                    tuple(h.shape),
                )

            del q, h, q_inv, h_inv
            if device.type == "cuda":
                torch.cuda.empty_cache()

        logging.info(
            "KFACStore ready: layers=%d, memory=%.4f GB, time=%.2f s",
            len(self.stats),
            self.memory_gb(),
            time.perf_counter() - t_total,
        )

    @staticmethod
    def _resolve_precompute_device(precompute_device: str) -> torch.device:
        if precompute_device == "cpu":
            return torch.device("cpu")
        if precompute_device == "cuda":
            if not torch.cuda.is_available():
                warnings.warn(
                    "kfac_precompute_device='cuda' requested without CUDA; "
                    "falling back to CPU.",
                    RuntimeWarning,
                )
                return torch.device("cpu")
            return torch.device("cuda", torch.cuda.current_device())

        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @staticmethod
    def _unwrap_factor(x: Any) -> Tensor:
        if isinstance(x, (list, tuple)):
            if not x:
                raise ValueError("Empty KFAC factor list.")
            x = x[-1]
        if not torch.is_tensor(x):
            raise TypeError(f"Expected KFAC factor Tensor, got {type(x)}.")
        return x

    @staticmethod
    def _damped_inverse(mat: Tensor, damping: float) -> Tensor:
        """Invert a symmetrized, damped KFAC factor in FP32."""
        mat = mat.detach().to(dtype=torch.float32)
        mat = 0.5 * (mat + mat.t())
        eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat_damped = mat + float(damping) * eye

        try:
            chol = torch.linalg.cholesky(mat_damped)
            return torch.cholesky_inverse(chol).contiguous()
        except RuntimeError:
            # Same damped matrix; this is only a numerical fallback.
            logging.warning(
                "Cholesky failed for KFAC factor %s; using torch.linalg.inv.",
                tuple(mat.shape),
            )
            return torch.linalg.inv(mat_damped).contiguous()

    def _lookup(self, layer_name: str) -> dict[str, Tensor]:
        candidates = [
            layer_name,
            layer_name + ".weight",
            layer_name.removesuffix(".weight"),
        ]
        for key in candidates:
            if key in self.stats:
                return self.stats[key]
        raise KeyError(
            f"No KFAC entry found for layer '{layer_name}'. Tried {candidates}."
        )

    def precondition(self, layer_name: str, grad: Tensor) -> Tensor:
        """Return H^{-1} G Q^{-1} for CGaLore basis selection."""
        if grad.ndim != 2:
            raise ValueError(
                "KFAC preconditioning expects a 2D weight gradient, "
                f"got {tuple(grad.shape)} for {layer_name}."
            )

        entry = self._lookup(layer_name)
        d_out, d_in = grad.shape
        device = grad.device

        q_inv = entry["Q_inv"].to(device=device, dtype=torch.float32)
        h_inv = entry["H_inv"].to(device=device, dtype=torch.float32)

        if q_inv.shape != (d_in, d_in):
            raise ValueError(
                f"KFAC Q_inv shape mismatch for {layer_name}: expected "
                f"{(d_in, d_in)}, got {tuple(q_inv.shape)}."
            )
        if h_inv.shape != (d_out, d_out):
            raise ValueError(
                f"KFAC H_inv shape mismatch for {layer_name}: expected "
                f"{(d_out, d_out)}, got {tuple(h_inv.shape)}."
            )

        g = grad.detach().to(device=device, dtype=torch.float32)
        return h_inv.matmul(g).matmul(q_inv)

    def memory_gb(self) -> float:
        total_bytes = 0
        for entry in self.stats.values():
            for value in entry.values():
                total_bytes += value.numel() * value.element_size()
        return total_bytes / 1024**3


class GaLoreProjector:
    """Fixed-rank GaLore projector used in the paper experiments.

    proj_type:
        "full":  P^T G Q       -> r x r
        "left":  P^T G         -> r x d_in
        "right": G Q           -> d_out x r

    Projection bases are refreshed every ``update_proj_gap`` optimizer steps
    using randomized SVD.
    """

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "full",
        svd_oversampling: int = 8,
        svd_niter: int = 1,
        report_time_fn: Callable[[str, float], None] | None = None,
        report_count_fn: Callable[[str, int], None] | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if update_proj_gap <= 0:
            raise ValueError(
                f"update_proj_gap must be positive, got {update_proj_gap}."
            )
        if proj_type not in {"left", "right", "full"}:
            raise ValueError(
                f"proj_type must be one of left/right/full, got {proj_type}."
            )
        if svd_oversampling < 0:
            raise ValueError(
                f"svd_oversampling must be non-negative, got {svd_oversampling}."
            )
        if svd_niter < 0:
            raise ValueError(f"svd_niter must be non-negative, got {svd_niter}.")

        self.rank = int(rank)
        self.update_proj_gap = int(update_proj_gap)
        self.scale = float(scale)
        self.proj_type = proj_type
        self.svd_oversampling = int(svd_oversampling)
        self.svd_niter = int(svd_niter)

        self.P: Tensor | None = None
        self.Q: Tensor | None = None

        self.report_time_fn = report_time_fn
        self.report_count_fn = report_count_fn

    @staticmethod
    def _cuda_sync_if_needed() -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def _time_block(self, key: str, fn):
        self._cuda_sync_if_needed()
        t0 = time.perf_counter()
        out = fn()
        self._cuda_sync_if_needed()
        elapsed = time.perf_counter() - t0
        if self.report_time_fn is not None:
            self.report_time_fn(key, elapsed)
        return out

    def _count(self, key: str, value: int = 1) -> None:
        if self.report_count_fn is not None:
            self.report_count_fn(key, value)

    def basis_matrix(self, grad: Tensor) -> Tensor:
        return grad.detach().to(dtype=torch.float32)

    def _basis_time_key(self) -> str:
        return "galore_basis_time"

    def should_refresh(self, step: int) -> bool:
        if self.P is None and self.Q is None:
            return True
        return step % self.update_proj_gap == 0

    @staticmethod
    def _randomized_svd(
        mat: Tensor,
        rank: int,
        oversampling: int = 8,
        niter: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Approximate truncated SVD using randomized range finding."""
        m, n = mat.shape
        r = min(rank, m, n)
        q = min(r + oversampling, m, n)

        # If the requested sketch spans the smaller matrix dimension, use the
        # exact decomposition; this is mathematically equivalent to requesting
        # the full randomized subspace and avoids unnecessary randomness.
        if q >= min(m, n):
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            return u[:, :r], s[:r], vh[:r, :]

        omega = torch.randn(n, q, device=mat.device, dtype=mat.dtype)
        y = mat.matmul(omega)
        for _ in range(niter):
            y = mat.matmul(mat.t().matmul(y))

        q_mat, _ = torch.linalg.qr(y, mode="reduced")
        b = q_mat.t().matmul(mat)
        u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
        u = q_mat.matmul(u_hat)

        return (
            u[:, :r].contiguous(),
            s[:r].contiguous(),
            vh[:r, :].contiguous(),
        )

    def refresh(self, grad: Tensor) -> None:
        mat = self._time_block(
            self._basis_time_key(),
            lambda: self.basis_matrix(grad),
        )
        if mat.ndim != 2:
            raise ValueError(
                f"GaLoreProjector expects a 2D tensor, got {tuple(mat.shape)}."
            )

        u, _, vh = self._time_block(
            "galore_svd_time",
            lambda: self._randomized_svd(
                mat,
                rank=self.rank,
                oversampling=self.svd_oversampling,
                niter=self.svd_niter,
            ),
        )
        self._count("galore_refresh_count", 1)

        r = min(self.rank, u.shape[1], vh.shape[0])
        if self.proj_type == "full":
            self.P = u[:, :r].contiguous()
            self.Q = vh[:r, :].t().contiguous()
        elif self.proj_type == "left":
            self.P = u[:, :r].contiguous()
            self.Q = None
        else:  # right
            self.P = None
            self.Q = vh[:r, :].t().contiguous()

    def project(self, grad: Tensor, step: int) -> Tensor:
        if self.should_refresh(step):
            self.refresh(grad)

        g = grad.detach().to(dtype=torch.float32)
        if self.proj_type == "full":
            assert self.P is not None and self.Q is not None
            return self.P.t().matmul(g).matmul(self.Q)
        if self.proj_type == "left":
            assert self.P is not None
            return self.P.t().matmul(g)

        assert self.Q is not None
        return g.matmul(self.Q)

    def project_back(
        self,
        low_rank_update: Tensor,
        target_dtype: torch.dtype,
    ) -> Tensor:
        if self.proj_type == "full":
            assert self.P is not None and self.Q is not None
            update = self.P.matmul(low_rank_update).matmul(self.Q.t())
        elif self.proj_type == "left":
            assert self.P is not None
            update = self.P.matmul(low_rank_update)
        else:  # right
            assert self.Q is not None
            update = low_rank_update.matmul(self.Q.t())

        return (update * self.scale).to(dtype=target_dtype)


class KFACGaLoreProjector(GaLoreProjector):
    """CGaLore projector: select bases from H^{-1} G Q^{-1}."""

    def __init__(
        self,
        rank: int,
        update_proj_gap: int,
        scale: float,
        kfac_store: KFACStore,
        layer_name: str,
        proj_type: str = "full",
        svd_oversampling: int = 8,
        svd_niter: int = 1,
        report_time_fn: Callable[[str, float], None] | None = None,
        report_count_fn: Callable[[str, int], None] | None = None,
    ) -> None:
        super().__init__(
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
            svd_oversampling=svd_oversampling,
            svd_niter=svd_niter,
            report_time_fn=report_time_fn,
            report_count_fn=report_count_fn,
        )
        self.kfac_store = kfac_store
        self.layer_name = layer_name

    def basis_matrix(self, grad: Tensor) -> Tensor:
        return self.kfac_store.precondition(self.layer_name, grad)

    def _basis_time_key(self) -> str:
        return "cgalore_kfac_time"


class GaLoreAdamW(NamedOptimizer, Optimizer):
    """AdamW with GaLore or curvature-guided GaLore projection."""

    def __init__(
        self,
        named_params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        rank=20,
        update_proj_gap=200,
        scale=1.0,
        use_cl_galore=False,
        kfac_path=None,
        kfac_damping=1e-3,
        kfac_precompute_device="auto",
        kfac_log_every=1,
        proj_type="full",
        svd_oversampling=8,
        svd_niter=1,
    ):
        param_groups = self._build_param_groups(
            named_params=named_params,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            use_cl_galore=use_cl_galore,
            proj_type=proj_type,
            svd_oversampling=svd_oversampling,
            svd_niter=svd_niter,
        )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._report_timers: dict[str, float] = {}
        self._report_counts: dict[str, int] = {}

        torch.optim.Optimizer.__init__(self, param_groups, defaults)
        self._log_galore_summary()

        galore_layer_names = {
            str(group["layer_name"])
            for group in self.param_groups
            if bool(group.get("use_galore", False)) and "layer_name" in group
        }

        self.kfac_store = (
            KFACStore(
                kfac_path,
                damping=kfac_damping,
                precompute_device=kfac_precompute_device,
                log_every=kfac_log_every,
                allowed_layer_names=galore_layer_names,
            )
            if kfac_path is not None
            else None
        )

        if use_cl_galore and self.kfac_store is None:
            warnings.warn(
                "use_cl_galore=True but kfac_path=None; falling back to "
                "vanilla GaLore.",
                RuntimeWarning,
            )

    def _build_param_groups(
        self,
        named_params,
        *,
        weight_decay: float,
        rank: int,
        update_proj_gap: int,
        scale: float,
        use_cl_galore: bool,
        proj_type: str,
        svd_oversampling: int,
        svd_niter: int,
    ):
        param_groups = []
        normal_decay_params = []
        normal_no_decay_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue

            if self._is_galore_param(name, param):
                param_groups.append(
                    {
                        "params": [param],
                        "use_galore": True,
                        "use_cl_galore": use_cl_galore,
                        "param_name": name,
                        "layer_name": name.removesuffix(".weight"),
                        "rank": rank,
                        "update_proj_gap": update_proj_gap,
                        "scale": scale,
                        "weight_decay": weight_decay,
                        "proj_type": proj_type,
                        "svd_oversampling": svd_oversampling,
                        "svd_niter": svd_niter,
                    }
                )
            elif self._is_no_weight_decay_param(name, param):
                normal_no_decay_params.append(param)
            else:
                normal_decay_params.append(param)

        if normal_decay_params:
            param_groups.append(
                {
                    "params": normal_decay_params,
                    "use_galore": False,
                    "weight_decay": weight_decay,
                }
            )
        if normal_no_decay_params:
            param_groups.append(
                {
                    "params": normal_no_decay_params,
                    "use_galore": False,
                    "weight_decay": 0.0,
                }
            )

        return param_groups

    def get_report_stats(self, reset: bool = True) -> dict[str, float]:
        stats = {
            "galore_projected_coords": float(
                getattr(self, "_galore_projected_coords", 0)
            ),
            "galore_full_trainable_params": float(
                getattr(self, "_galore_full_trainable_params", 0)
            ),
            "galore_normal_trainable_params": float(
                getattr(self, "_galore_normal_trainable_params", 0)
            ),
        }

        if self.kfac_store is not None:
            stats["kfac_cpu_mem_GB"] = self.kfac_store.memory_gb()

        for key, value in self._report_timers.items():
            stats[key] = float(value)
        for key, value in self._report_counts.items():
            stats[key] = float(value)

        if reset:
            self._report_timers.clear()
            self._report_counts.clear()
        return stats

    @staticmethod
    def _contains_any(name: str, patterns) -> bool:
        name = name.lower()
        return any(pattern.lower() in name for pattern in patterns)

    @staticmethod
    def _cuda_sync_if_needed() -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def _add_report_time(self, key: str, value: float) -> None:
        self._report_timers[key] = self._report_timers.get(key, 0.0) + float(value)

    def _add_report_count(self, key: str, value: int = 1) -> None:
        self._report_counts[key] = self._report_counts.get(key, 0) + int(value)

    def _is_galore_param(self, name: str, param) -> bool:
        if param.ndim != 2 or not name.endswith(".weight"):
            return False

        # The paper updates linear weight matrices but freezes output layers.
        excludes = [
            "bias",
            "norm",
            "layer_norm",
            "batch_norm",
            "embedding",
            "ctc",
            "output",
            "lm_head",
            "decoder.output",
            "classifier",
        ]
        return not self._contains_any(name, excludes)

    def _is_no_weight_decay_param(self, name: str, param) -> bool:
        if param.ndim == 1 or name.endswith(".bias"):
            return True
        return self._contains_any(
            name,
            ["bias", "norm", "layer_norm", "batch_norm", "embedding"],
        )

    def _make_projector(self, group: dict[str, Any]) -> GaLoreProjector:
        rank = group.get("rank")
        if rank is None:
            raise ValueError("GaLore parameter group requires 'rank'.")

        kwargs = dict(
            rank=int(rank),
            update_proj_gap=int(group.get("update_proj_gap", 200)),
            scale=float(group.get("scale", 1.0)),
            proj_type=str(group.get("proj_type", "full")),
            svd_oversampling=int(group.get("svd_oversampling", 8)),
            svd_niter=int(group.get("svd_niter", 1)),
            report_time_fn=self._add_report_time,
            report_count_fn=self._add_report_count,
        )

        if bool(group.get("use_cl_galore", False)) and self.kfac_store is not None:
            layer_name = group.get("layer_name")
            if layer_name is None:
                raise ValueError("CGaLore requires param-group field 'layer_name'.")
            return KFACGaLoreProjector(
                kfac_store=self.kfac_store,
                layer_name=str(layer_name),
                **kwargs,
            )

        return GaLoreProjector(**kwargs)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group.get("maximize", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients.")
                if maximize:
                    grad = -grad

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                step_num = int(state["step"])
                use_galore = bool(group.get("use_galore", False)) and p.ndim == 2

                if use_galore:
                    if "projector" not in state:
                        state["projector"] = self._make_projector(group)
                    projector = state["projector"]
                    assert isinstance(projector, GaLoreProjector)

                    self._cuda_sync_if_needed()
                    t0 = time.perf_counter()
                    grad_for_adam = projector.project(grad, step=step_num)
                    self._cuda_sync_if_needed()
                    self._add_report_time(
                        "galore_project_time", time.perf_counter() - t0
                    )
                else:
                    projector = None
                    grad_for_adam = grad.detach().to(dtype=torch.float32)

                if (
                    "exp_avg" not in state
                    or state["exp_avg"].shape != grad_for_adam.shape
                ):
                    state["exp_avg"] = torch.zeros_like(grad_for_adam)
                    state["exp_avg_sq"] = torch.zeros_like(grad_for_adam)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = int(state["step"])

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad_for_adam, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_for_adam,
                    grad_for_adam,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom

                if use_galore:
                    assert projector is not None
                    self._cuda_sync_if_needed()
                    t0 = time.perf_counter()
                    full_update = projector.project_back(
                        update,
                        target_dtype=p.dtype,
                    )
                    self._cuda_sync_if_needed()
                    self._add_report_time(
                        "galore_project_back_time", time.perf_counter() - t0
                    )
                else:
                    full_update = update.to(dtype=p.dtype)

                p.add_(full_update, alpha=-step_size)

        return loss

    def _log_galore_summary(self) -> None:
        galore_layers = 0
        galore_full_params = 0
        galore_projected_coords = 0
        normal_trainable_params = 0

        for group in self.param_groups:
            use_galore = bool(group.get("use_galore", False))
            rank = int(group.get("rank", 0))
            proj_type = str(group.get("proj_type", "full"))

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if use_galore and p.ndim == 2:
                    galore_layers += 1
                    d_out, d_in = p.shape
                    r = min(rank, d_out, d_in)
                    galore_full_params += p.numel()

                    if proj_type == "full":
                        coords = r * r
                    elif proj_type == "left":
                        coords = r * d_in
                    elif proj_type == "right":
                        coords = d_out * r
                    else:
                        raise ValueError(f"Unknown proj_type: {proj_type}")
                    galore_projected_coords += coords
                else:
                    normal_trainable_params += p.numel()

        total_optimizer_state_coords = (
            2 * galore_projected_coords + 2 * normal_trainable_params
        )

        logging.info(
            "GaLore summary: layers=%d, full_params=%d, projected_coords=%d, "
            "normal_params=%d, Adam_state_coords=%d",
            galore_layers,
            galore_full_params,
            galore_projected_coords,
            normal_trainable_params,
            total_optimizer_state_coords,
        )

        self._galore_layers = galore_layers
        self._galore_full_trainable_params = galore_full_params
        self._galore_projected_coords = galore_projected_coords
        self._galore_normal_trainable_params = normal_trainable_params
        self._galore_adam_state_coords_estimate = total_optimizer_state_coords


def build_galore_param_groups(
    model: torch.nn.Module,
    optim_conf: dict[str, Any],
    exclude_weight_decay_conf: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility helper for ESPnet setups using external param groups."""
    from espnet2.optimizers.optim_groups import add_optimizer_hooks

    rank = optim_conf["rank"]
    update_proj_gap = optim_conf.get("update_proj_gap", optim_conf.get("T", 200))
    scale = optim_conf.get("scale", 1.0)
    use_cl_galore = optim_conf.get("use_cl_galore", False)
    weight_decay = optim_conf.get("weight_decay", 0.0)
    proj_type = optim_conf.get("proj_type", "full")
    svd_oversampling = optim_conf.get("svd_oversampling", 8)
    svd_niter = optim_conf.get("svd_niter", 1)

    if exclude_weight_decay_conf is not None:
        add_optimizer_hooks(model, **exclude_weight_decay_conf)

    galore_param_ids: set[int] = set()
    param_groups: list[dict[str, Any]] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        weight = module.weight
        if weight is None or not weight.requires_grad:
            continue

        lower_name = module_name.lower()
        if any(
            pattern in lower_name
            for pattern in ("ctc", "output", "lm_head", "decoder.output", "classifier")
        ):
            continue

        group_weight_decay = weight_decay
        if hasattr(weight, "_optim"):
            group_weight_decay = getattr(weight, "_optim").get(
                "weight_decay", group_weight_decay
            )

        param_groups.append(
            {
                "params": [weight],
                "use_galore": True,
                "use_cl_galore": use_cl_galore,
                "layer_name": module_name,
                "rank": rank,
                "update_proj_gap": update_proj_gap,
                "scale": scale,
                "weight_decay": group_weight_decay,
                "proj_type": proj_type,
                "svd_oversampling": svd_oversampling,
                "svd_niter": svd_niter,
            }
        )
        galore_param_ids.add(id(weight))

    other_decay_params: list[Tensor] = []
    other_no_decay_params: list[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in galore_param_ids:
            continue

        if param.ndim == 1 or name.endswith(".bias"):
            other_no_decay_params.append(param)
        elif hasattr(param, "_optim"):
            param_weight_decay = getattr(param, "_optim").get(
                "weight_decay", weight_decay
            )
            if param_weight_decay == 0.0:
                other_no_decay_params.append(param)
            else:
                other_decay_params.append(param)
        else:
            other_decay_params.append(param)

    if other_decay_params:
        param_groups.append(
            {
                "params": other_decay_params,
                "use_galore": False,
                "weight_decay": weight_decay,
            }
        )
    if other_no_decay_params:
        param_groups.append(
            {
                "params": other_no_decay_params,
                "use_galore": False,
                "weight_decay": 0.0,
            }
        )

    return param_groups
