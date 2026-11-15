# Figure QA notes

The final single-panel figure was visually inspected: the final checkpoint has the strongest visual emphasis, all five curves and four observations per curve are shown, including the baseline nonmonotonic segment. The legend, axes and source qualification are readable and do not overlap. One panel makes comparative panel alignment not applicable. The rendered PDF glyph minimum is 7 pt, and the PDF collision audit has 0 failures and 0 warnings.

Source preflight: 0 failures. Reviewed warnings: PNG is the requested report preview rather than a TIFF submission raster; positivity is asserted for every actual-byte value before applying log scaling; the three related same-scene holdouts have no justified seed confidence interval, so the arithmetic mean is explicitly descriptive and all individual observations remain in CSV. No error bars or significance claims are invented.

The first source audit could not infer dynamically formed export suffixes despite actual PDF/SVG/PNG files being present. Export calls were made explicit and the source audit rerun; its first report is retained. The final data/layout were re-rendered and PDF text/collision checks rerun after that change. No publication-submission compliance claim is made.
