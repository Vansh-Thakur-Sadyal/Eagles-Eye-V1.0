"""Eagles Eye agent roster."""
from .base import Agent, CameraContext, Finding, FrameContext
from .behavior import BehaviorAgent
from .commander import IncidentCommander
from .crowd import CrowdAgent
from .identity import AppearanceAgent, OcclusionAgent, ReIDAgent
from .objects import ObjectAgent
from .orchestrator import Orchestrator, get_orchestrator
from .privacy import PrivacyAgent
from .relationship import RelationshipAgent
from .spatial import SpatialAgent
from .threat import ThreatAssessment, ThreatAssessor
from .watchlist import EnrolledSubject, WatchlistAgent

__all__ = [
    "Agent", "CameraContext", "Finding", "FrameContext",
    "BehaviorAgent", "IncidentCommander", "CrowdAgent",
    "AppearanceAgent", "OcclusionAgent", "ReIDAgent",
    "ObjectAgent", "Orchestrator", "get_orchestrator",
    "PrivacyAgent", "RelationshipAgent", "SpatialAgent",
    "ThreatAssessment", "ThreatAssessor",
    "EnrolledSubject", "WatchlistAgent",
]
