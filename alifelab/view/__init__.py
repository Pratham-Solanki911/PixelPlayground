"""alifelab.view -- engine-free interactive visualizer package.

Everything in this package consumes only plain dataclasses + NumPy arrays
(:mod:`alifelab.view.model`). It never imports the simulation, RL code,
Trainer, or Agent, so it can be driven by a live session, a saved recording,
or a scripted demo interchangeably.
"""

from .adapter import (
    SnapshotLike,
    entity_to_view,
    ephemera_from_diagnostics,
    event_to_view,
    snapshot_to_frame,
)
from .controller import PLAY_SPEEDS, PlaybackController
from .debug import debug_overview, debug_report, debug_tests
from .events import EventLog, Genealogy, LogEntry
from .gui import (
    DashboardPanel,
    EventPanel,
    InspectorPanel,
    LineagePanel,
    Panel,
    TextPanel,
    stack_horizontal,
    stack_vertical,
)
from .inspector import Inspector, InspectorField, InspectorReport
from .model import (
    TERRAIN_EMPTY,
    TERRAIN_WALL,
    TERRAIN_WATER,
    AgentEphemera,
    ViewEntity,
    ViewEvent,
    ViewFrame,
    ViewWorld,
)
from .population import PopulationTracker
from .recording import Recorder
from .renderers import (
    EntityRenderer,
    ObservationRenderer,
    OverlayRenderer,
    Renderer,
    WorldRenderer,
)
from .sources import FrameSource, SnapshotSource, StaticFrameSource
from .visualizer import HELP_LINES, MODES, Visualizer

__all__ = [
    "AgentEphemera",
    "DashboardPanel",
    "EntityRenderer",
    "EventLog",
    "EventPanel",
    "FrameSource",
    "Genealogy",
    "HELP_LINES",
    "Inspector",
    "InspectorField",
    "InspectorPanel",
    "InspectorReport",
    "LineagePanel",
    "LogEntry",
    "MODES",
    "ObservationRenderer",
    "OverlayRenderer",
    "Panel",
    "PLAY_SPEEDS",
    "PlaybackController",
    "PopulationTracker",
    "Renderer",
    "Recorder",
    "SnapshotLike",
    "SnapshotSource",
    "StaticFrameSource",
    "TERRAIN_EMPTY",
    "TERRAIN_WALL",
    "TERRAIN_WATER",
    "TextPanel",
    "ViewEntity",
    "ViewEvent",
    "ViewFrame",
    "ViewWorld",
    "Visualizer",
    "WorldRenderer",
    "debug_overview",
    "debug_report",
    "debug_tests",
    "entity_to_view",
    "ephemera_from_diagnostics",
    "event_to_view",
    "snapshot_to_frame",
    "stack_horizontal",
    "stack_vertical",
]