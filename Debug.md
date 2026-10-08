# Debugging the Visualizer

Everything in `alifelab/view` is engine-free: it runs on machines with no
simulation, no RL code, no pygame, no torch. That makes its debug harness run
literally anywhere Python can import the package.

## Self-tests (`V` in the GUI, or directly)

Run the built-in self-tests from Python (no display needed):

```python
from alifelab.view import debug_tests
for line in debug_tests(verbose=True):
    print(line)
```

Or from the shell:

```bash
python -c "from alifelab.view import debug_tests; print('\n'.join(debug_tests(verbose=True)))"
python -m unittest discover -s tests -v
python _smoke_view.py
```

The tests cover five categories and must all pass before any view change is
considered safe:

| Category   | Checks                                                                 |
|------------|------------------------------------------------------------------------|
| `model`    | world/entity/frame construction, by-id lookup, display names           |
| `renderers`| world+overlay+observation renderers produce correctly shaped images     |
| `events`   | log formatting (known + unknown kinds), genealogy, population, inspector|
| `adapters` | snapshot→frame conversion, playback speeds, recorder round-trip         |
| `_smoke_view.py` | full headless visualizer run + key bindings + recording + report  |

## Debug overview (`G` in the GUI)

`debug_overview(visualizer)` prints a one-screen digest: mode, episode/step,
playback state, world size, live populations, cumulative counters, focus
state, panel/overlay toggles, recording state, and the last debug-report path.

## Debug report file (`D` in the GUI)

`visualizer.save_debug_report()` writes the overview to
`runs/view/debug_<episode>_<step>.txt` and records the path in
`visualizer.report_path`. Use it to attach concrete state to a bug report.

## Common failure modes

- `ImportError: no module named 'pygame'` — the GUI loop needs `pip install
  alifelab[ui]`; headless `run(headless=True)` and all selftests work without it.
- Different behavior between machines — never richer: the view only renders
  plain dataclasses + arrays, so any reproduction driven by a saved `npz`
  recording reproduces pixel-for-pixel.
- A rule emits an event kind that renders as
  `weird: actor=... target=...` — that's the generic fallback logging every
  unknown kind. Nothing is silently dropped.