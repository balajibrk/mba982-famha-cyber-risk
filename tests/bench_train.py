"""Quick benchmark: time one 20-epoch fold on CPU vs CUDA to pick the faster device."""
import time

import torch
import torch.nn.functional as F

from src import config as C
from src.model.features import D_IN, load_dataset
from src.model.xfamha_gnn import XFAMHAGNN


def bench(device_str, epochs=20):
    device = torch.device(device_str)
    ds = load_dataset()
    slugs = [c.slug for c in C.COMPANY_LIST]
    train = [ds[s] for s in slugs[1:]]
    model = XFAMHAGNN(d_in=D_IN, d_model=32, n_layers=3).to(device)
    model.build_structure([{"x": s["x"]} for s in train])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    # move sample tensors once
    cache = [(s["x"].to(device), s["edge_index"].to(device),
              torch.tensor([s["y"]], device=device)) for s in train]
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        total = 0.0
        for x, ei, y in cache:
            total = total + F.cross_entropy(model(x, ei), y)
        (total / len(cache)).backward()
        opt.step()
    if device_str == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"{device_str:5s}: {epochs} epochs x {len(cache)} graphs = {dt:.2f}s "
          f"({dt/epochs*1000:.1f} ms/epoch)")
    return dt


if __name__ == "__main__":
    bench("cpu")
    if torch.cuda.is_available():
        bench("cuda")
