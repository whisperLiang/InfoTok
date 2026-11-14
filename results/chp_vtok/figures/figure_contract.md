# Figure contract

Question: do more transmitted bytes improve the early CHP reconstruction, and does the multiscale variant improve matched-byte quality?

Core inference: report the observed behavior, including negative incremental gains and inconsistent multiscale benefits; do not infer convergence or generalization. Three short clips were also used for smoke training. No held-out benchmark or repeated-seed uncertainty claim.

Main quantitative figure: PSNR supplies pixel fidelity, SSIM local structure, LPIPS and DISTS perceptual distortion. Individual video curves are retained, with descriptive means only. No monotonic fitting, best-prefix clipping, synthetic points or confidence intervals.

Paired ablation: subtract single-scale from multiscale only after matching exact video, budget and actual bytes. Anchor-order ablation uses a separately trained Morton checkpoint under the same smoke curriculum, with geometry diagnostics kept separate from reconstruction quality.

Exports: Python/matplotlib only; editable PDF/SVG, PNG preview; 180 mm quantitative grid width; font floor 7 pt. Final plot rectangles are audited to 1.5 pt, then source, PDF text and collisions audited. Reconstructed images come from received integer IDs; contact-sheet resizing is display only, and full-resolution frames/tensors remain available.
