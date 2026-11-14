"""Deterministic rectangular-grid anchors, independent of transmitted budget."""

from collections import deque
from functools import lru_cache

import numpy as np
import torch


def radical_inverse(index, base):
    value, scale = 0.0, 1.0
    while index:
        index, digit = divmod(index, base)
        scale /= base
        value += digit * scale
    return value


def _halton_cells(height, width):
    cells, visited, index = [], set(), 1
    while len(cells) < height * width:
        cell = (
            min(height - 1, int(radical_inverse(index, 2) * height)),
            min(width - 1, int(radical_inverse(index, 3) * width)),
        )
        if cell not in visited:
            visited.add(cell)
            cells.append(cell)
        index += 1
    return tuple(cells)


def _subrectangles(y0, y1, x0, x1):
    if y1 - y0 == 1 and x1 - x0 == 1:
        return []
    ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
    ys = [(y0, ym), (ym, y1)] if y1 - y0 > 1 else [(y0, y1)]
    xs = [(x0, xm), (xm, x1)] if x1 - x0 > 1 else [(x0, x1)]
    return [(a, b, c, d) for a, b in ys for c, d in xs]


def _morton_cells(height, width):
    cells, visited = [], set()
    queue = deque([(0, height, 0, width)])
    while queue:
        rectangle = queue.popleft()
        y0, y1, x0, x1 = rectangle
        cell = ((y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2)
        if cell not in visited:
            cells.append(cell)
            visited.add(cell)
        queue.extend(_subrectangles(*rectangle))
    return tuple(cells)


@lru_cache(maxsize=64)
def grid_order(height, width, order="halton"):
    if min(height, width) < 1:
        raise ValueError("empty anchor grid")
    functions = {"halton": _halton_cells, "morton": _morton_cells}
    if order not in functions:
        raise ValueError("unknown anchor ordering")
    return functions[order](height, width)


def anchors(height, width, count, order="halton", device=None):
    cells = grid_order(height, width, order)
    # Only tiny debug grids can require multiple distinct learned slots per cell.
    coords = torch.tensor([cells[i % len(cells)] for i in range(count)], device=device)
    normalized = (coords.float() + 0.5) / torch.tensor([height, width], device=device)
    return coords, normalized


def coverage(height, width, count, order="halton", partitions=8, radius=2):
    cells, normalized = anchors(height, width, count, order)
    grid = torch.stack(
        torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij"), -1
    ).float()
    # Chunk distances to avoid a full high-resolution all-pairs matrix.
    minimum = torch.cat(
        [torch.cdist(g, cells.float()).amin(1) for g in grid.reshape(-1, 2).split(1024)]
    )
    infinity_distance = torch.cat(
        [
            (g[:, None] - cells.float()[None]).abs().amax(-1).amin(-1)
            for g in grid.reshape(-1, 2).split(512)
        ]
    )
    bins = (normalized * partitions).long().clamp(max=partitions - 1)
    histogram = np.zeros((partitions, partitions), dtype=int)
    np.add.at(histogram, (bins[:, 0].numpy(), bins[:, 1].numpy()), 1)
    return dict(
        height=height,
        width=width,
        count=count,
        order=order,
        unique_cells=int(torch.unique(cells, dim=0).shape[0]),
        grid_fraction=float(torch.unique(cells, dim=0).shape[0] / (height * width)),
        max_uncovered_distance_cells=float(minimum.max()),
        local_window_radius=radius,
        local_window_covered_fraction=float(
            (infinity_distance <= radius).float().mean()
        ),
        covered_partition_fraction=float((histogram > 0).mean()),
        partition_counts=histogram.tolist(),
    )
