"""Phase 3 validation: FAMHA head determination, param-count inequality (Thm 3.1),
and a full GPU forward/backward on a real company KG."""
import sys

import numpy as np
import torch

from src import config as C
from src.model.famha import FAMHA
from src.model.features import load_dataset, D_IN
from src.model.xfamha_gnn import XFAMHAGNN

torch.manual_seed(C.RANDOM_SEED)
np.random.seed(C.RANDOM_SEED)


def _synth(d: int, n: int, spread: str) -> torch.Tensor:
    """Random n x d matrix with either a concentrated or spread eigen-spectrum."""
    if spread == "concentrated":
        # variance dominated by 2 directions -> few eigenvalues above mean
        stds = np.array([8.0, 6.0] + [0.05] * (d - 2))
    else:  # "spread": many comparable directions -> more eigenvalues above mean
        stds = np.linspace(1.0, 2.0, d)
    X = np.random.randn(n, d) * stds[None, :]
    return torch.from_numpy(X.astype(np.float32))


def main() -> int:
    checks = []

    # --- unit test 1: head count responds to eigenvalue spread ------------- #
    d, n = 24, 400
    h_conc = FAMHA.determine_num_heads(_synth(d, n, "concentrated"))
    h_spread = FAMHA.determine_num_heads(_synth(d, n, "spread"))
    checks.append((
        f"head count rises with eigenvalue spread (concentrated={h_conc} < spread={h_spread})",
        h_conc < h_spread, f"conc={h_conc} spread={h_spread}",
    ))

    # --- unit test 2: FAMHA param count < vanilla MHA (Theorem 3.1 / Eq. 6) - #
    fam = FAMHA(d_model=d)
    fam.fit_structure(_synth(d, n, "spread"))
    theta_famha = fam.param_count()
    theta_normal = fam.vanilla_param_count()
    lens_sq = sum(li * li for li in fam.lens)
    checks.append((
        f"theta_FAMHA ({theta_famha}) < theta_normal ({theta_normal}) [h={fam.h}, "
        f"sum(len_i)={sum(fam.lens)}=d={d}]",
        theta_famha < theta_normal and sum(fam.lens) == d,
        f"lens={fam.lens} sum_sq={lens_sq}",
    ))

    # --- unit test 3: full model forward+backward on a real KG, on GPU ------ #
    ds = load_dataset()
    device = C.get_device()
    model = XFAMHAGNN(d_in=D_IN, d_model=32, n_layers=3).to(device)
    samples = [{"x": ds[s]["x"]} for s in ds]
    model.build_structure(samples)

    demo = ds["uber"]
    x = demo["x"].to(device)
    ei = demo["edge_index"].to(device)
    y = torch.tensor([demo["y"]], device=device)

    logits = model(x, ei)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    on_cuda = next(model.parameters()).is_cuda

    checks.append((f"forward ok: logits shape {tuple(logits.shape)} == (1, 4)",
                   tuple(logits.shape) == (1, 4), str(tuple(logits.shape))))
    checks.append((f"backward ok: finite loss={loss.item():.4f}, grads flow",
                   torch.isfinite(loss).item() and has_grad, ""))
    checks.append((f"model runs on GPU (is_cuda={on_cuda})",
                   on_cuda or not torch.cuda.is_available(), f"cuda_avail={torch.cuda.is_available()}"))

    fam_c, van_c = model.famha_param_counts()
    print(f"\nmodel head counts per layer: {model.head_counts()}")
    print(f"model total FAMHA params={fam_c} vs vanilla MHA-equivalent={van_c}")

    allpass = all(ok for _, ok, _ in checks)
    print()
    for name, ok, info in checks:
        print(("PASS" if ok else "FAIL"), "-", name, ("" if ok else ":: " + info))
    print("\nPHASE 3", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
