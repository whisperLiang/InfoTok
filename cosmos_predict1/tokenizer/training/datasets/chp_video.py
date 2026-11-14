"""Native-resolution frames; optional explicitly declared debug resize/crop only."""

from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CHPVideoDataset(Dataset):
    def __init__(
        self, video_pattern, num_video_frames=17, debug_height=None, crop_size=None
    ):
        self.paths = sorted(glob(video_pattern))
        if not self.paths:
            raise ValueError(f"no videos: {video_pattern}")
        if (num_video_frames - 1) % 4:
            raise ValueError("training clips require 1+4n frames")
        self.frames, self.debug_height, self.crop_size = (
            num_video_frames,
            debug_height,
            crop_size,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = int(torch.randint(max(1, total - self.frames + 1), ()).item())
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frames):
            ok, bgr = cap.read()
            if not ok:
                cap.release()
                raise ValueError(f"video shorter than training clip: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.debug_height is not None:
                h, w = rgb.shape[:2]
                rgb = cv2.resize(
                    rgb,
                    (round(w * self.debug_height / h), self.debug_height),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(rgb)
        cap.release()
        x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 127.5 - 1
        if self.crop_size is not None:
            size = self.crop_size
            h, w = x.shape[-2:]
            if min(h, w) < size:
                raise ValueError("native crop exceeds source geometry")
            y = int(torch.randint(h - size + 1, ()).item())
            z = int(torch.randint(w - size + 1, ()).item())
            x = x[..., y : y + size, z : z + size]
        return {"video": x, "source": path}
