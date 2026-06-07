This folder contains three short audio cases used by the GUI demo:

- `*_mixture.wav`: two-speaker mixture input.
- `*_target.wav`: clean target speaker reference for listening comparison.
- `*_ours.wav`: target extraction output used in the interactive demo.
- `*_baseline.wav`: early baseline output used for audible contrast.

The final quantitative results in the report are based on the shared-clean80
TD-SpeakerBeam experiments, not on these three illustrative demo clips alone.
The GUI text has been updated to match the final TD-SpeakerBeam report line:
reliable target speaker extraction, enrollment sanity checks, and embedding-level
multi-enrollment pooling.
