# Setup

Requires Python **3.11+**.

## Base install

```bash
pip install -e .
```

## Optional extras

| Extra        | Provides                                         |
|--------------|--------------------------------------------------|
| `alifelab[ui]`  | `pygame` — the interactive (windowed) visualizer |
| `alifelab[test]`| `pytest` — run the test suite with pytest        |

```bash
pip install -e ".[ui,test]"
```

## Verify

```bash
python -m unittest discover -s tests -v     # all view + engine-free tests
python _smoke_view.py                        # headless visualizer run (no pygame)
```

## Run the visualizer

Headless (any machine, no window, drives a FrameSource + full data panels):

```python
from alifelab.view import Visualizer, SnapshotSource
viz = Visualizer.build(SnapshotSource(session), headless=True)
viz.run(frames=100)   # or call viz.step_once() manually
```

Interactive (needs `pygame`):

```python
viz = Visualizer.build(SnapshotSource(session))
viz.run()
```

`Visualizer.build()` runs sanity assertions on the wiring before starting, so
misconfiguration fails fast with a clear message instead of mid-run.

## Wire recording

The visualizer re-uses the engine's existing renderers through providers, so
recording logic is never duplicated:

```python
from alifelab.render.rgb import RGBRenderer
from alifelab.render.ascii import AsciiRenderer

rgb = RGBRenderer(scale=...)
viz.set_rgb_provider(lambda: rgb.render(session.snapshot()))       # or viz.display_frame()
viz.set_ascii_provider(lambda: AsciiRenderer().render(session.snapshot()))
```