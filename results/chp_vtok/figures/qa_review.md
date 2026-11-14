# Figure verification

Final quality source: `quality_reference_fp32/rd_*/metrics.json`; 28 observed native 17-frame reconstructions. Old BF16-reference observations and old draft plots remain separate and are superseded.

- Python/matplotlib only. Source, PDF/SVG and 300 dpi PNG exports retained. Editable PDF/SVG text, minimum actual PDF font size 7 pt.
- Rate-distortion plot: final 2×2 rendered rectangles pass the 1.5 pt alignment gate; no exemptions. Each metric retains per-video curves and descriptive means, no invented seed CI or monotonic fit.
- Native anchor heatmap: final 2×4 rectangles pass; correct 1088:1920 aspect; anchor counts normalized by mean count per spatial bin, shared color range. This establishes geometry only.
- All four final quantitative PDFs pass text-size and collision checks with zero failures/warnings. Paired and anchor-order legends were moved outside plot regions after an earlier draft failed collision checks; draft evidence remains archived.
- Visual inspection covered each metric panel, both ablation curves and all eight anchor panels, plus real Original/256/512/1024/2048 reconstruction sheets. Low reconstruction quality is clearly visible and is not hidden or sharpened.
- Source-preflight warnings reviewed: PNG preview rather than TIFF is intentional for this engineering report, vector PDF/SVG are the export masters; 300 dpi meets the declared preview contract. Descriptive means of three related smoke clips have no independent training-seed uncertainty, so no unsupported error bars were added.
- Contact sheets resize only for display. Original-size PNG frames, complete RGB tensor reconstructions and native MP4 display exports remain available; scientific metrics use tensors, not MP4 or scaled contact sheets.
