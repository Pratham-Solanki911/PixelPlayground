# Tips

## The engine-free rule

`alifelab.view` consumes only plain dataclasses (`ViewFrame`, `ViewWorld`,
`ViewEntity`, `ViewEvent`, `AgentEphemera`) plus NumPy arrays. It never
imports the simulation, RL code, Trainer, or Agent. Consequences:

- Same rendered pixels whether the frame comes from a live run, a saved
  `npz` recording, or a scripted demo (`StaticFrameSource`).
- Selftests run on machines with no simulation code installed at all.
- The eventual driver only needs to produce `ViewFrame`s (via
  `snapshot_to_frame` / `ephemera_from_diagnostics`).

## Connection your session should expose

The live wrapper `SnapshotSource` drives any object that exposes:

```python
def snapshot(self) -> ...   # standardized world state
def step(self) -> None
def reset(self) -> None
def set_mode(self, mode: str) -> None
```

## Keyboard map

```
space  pause/play       right/s  step once      r  reset episode
f  fast-forward         e/c      speed up/down  m  cycle mode
1-9  focus agent        z  select focused agent x  clear selection
o  toggle header        p  toggle panels        h  help
g  debug overview       d  save debug report    l  save event log
t  rgb recording on/off b  save rgb npz         a  ascii recording
q/esc  quit
```

## New event kinds

Add a template line in `alifelab/view/events.py::_TEMPLATES` for a readable
message; until then the generic fallback already logs it as-is. Keep event
`data` payloads simple (strings/numbers) so fallback formatting stays generic.

## New species

Nothing to do. Colors are derived from the species name via a deterministic
hash, and renderers/palettes/panels all key off `species` (falling back to
`kind`). Conflicting colors can be pinned in
`alifelab/view/palette.py::SPECIES_COLORS`.

## Recording channels

- RGB → `runs/view/video_<episode>_<step>.npz` (frames key, `(N,H,W,3)` uint8).
- ASCII → `runs/view/log_<episode>_<step>.txt` (transcript of provider output).
- Event log → `runs/view/eventlog_...txt`.

Both recording channels are independent; they can run while paused, and they
produce identical output headless.