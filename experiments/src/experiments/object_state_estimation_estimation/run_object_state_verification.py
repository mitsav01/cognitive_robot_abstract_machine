#!/usr/bin/env python3
"""
VESPER: Action-Aware Multimodal Object State Verification Demonstration.
Demonstrates temporal memory bounding, duplicate suppression, in-transition
uncertainty elevation, and post-action visual settlement within the Semantic Digital Twin.
"""

import time
from unittest.mock import MagicMock

from semantic_digital_twin.world_description.world_entity import WorldEntity
from semantic_digital_twin.semantic_annotations.object_state import (
    FillLevel,
    FillState,
    IsFilled,
    IsEmpty,
)
from semantic_digital_twin.semantic_annotations.state_manager import (
    DynamicStateManager,
    LowConfidenceError,
)


def run_demonstration():
    print("=" * 75)
    print("VESPER: Dynamic State Verification & Temporal Memory Demonstration")
    print("=" * 75)

    # Disable wall-clock max_history_seconds cleanup for synthetic step offsets,
    # or keep it sufficiently large (e.g. None) so ring-buffer (maxlen) manages size.
    manager = DynamicStateManager(max_history_per_object=10, max_history_seconds=None)
    entity_id = "ceramic_mug_01"
    dummy_entity = MagicMock(spec=WorldEntity)

    # Base anchor timestamp using current time
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Phase 1: Initial Perception (Pre-Action)
    # -------------------------------------------------------------------------
    print("\n[Phase 1: Initial Perception]")
    state_empty = FillState(
        target=dummy_entity,
        state=FillLevel.EMPTY,
        confidence=0.94,
        source="SigLIP2_ZeroShot",
        timestamp=t0,
    )
    manager.update_state(entity_id, state_empty)
    print(
        f" -> Observation registered: {state_empty.state.value.upper()} (conf: {state_empty.confidence:.2f}) at t=0.0s")

    empty_pred = IsEmpty(manager, entity_id, min_confidence=0.8)
    filled_pred = IsFilled(manager, entity_id, min_confidence=0.8)
    print(f" -> Query IsEmpty():  {empty_pred()}")
    print(f" -> Query IsFilled(): {filled_pred()}")

    # -------------------------------------------------------------------------
    # Phase 2: Perceptual Flicker & Duplicate Suppression
    # -------------------------------------------------------------------------
    print("\n[Phase 2: Perceptual Flicker & Duplicate Suppression]")
    # High-frequency re-detection (0.1s later) with identical semantic state
    state_dup = FillState(
        target=dummy_entity,
        state=FillLevel.EMPTY,
        confidence=0.93,
        source="SigLIP2_ZeroShot",
        timestamp=t0 + 0.1,
    )
    manager.update_state(entity_id, state_dup)
    history = manager.get_state_history(entity_id)
    print(f" -> Consecutive identical state received. History length suppressed to: {len(history)} record(s)")

    # -------------------------------------------------------------------------
    # Phase 3: Active Action Phase (Occlusion / In-Transition)
    # -------------------------------------------------------------------------
    print("\n[Phase 3: Active Manipulation / Action Phase]")
    print(" -> Robot executes 'PouringAction' at t=2.0s. Gripper occludes aperture.")
    print(" -> Transition entropy elevated: Confidence drops below safety bounds.")

    t_action = t0 + 2.0
    state_trans = FillState(
        target=dummy_entity,
        state=FillLevel.FILLED,
        confidence=0.45,  # Occlusion causes noisy/unreliable belief
        source="VLM_Occluded",
        timestamp=t_action,
    )
    manager.update_state(entity_id, state_trans)

    # Attempt safety-critical symbolic query
    strict_filled_pred = IsFilled(manager, entity_id, min_confidence=0.8)
    try:
        strict_filled_pred()
        print(" [WARNING] State passed without safety validation!")
    except LowConfidenceError as e:
        print(f" -> [Safety Gate Triggered]: Downstream planner query rejected ({e})")

    # -------------------------------------------------------------------------
    # Phase 4: Post-Action Settlement & Visual Validation
    # -------------------------------------------------------------------------
    print("\n[Phase 4: Post-Action Visual Settlement]")
    print(" -> Robot arm retreats at t=5.0s; unoccluded visual access restored.")
    t_settled = t0 + 5.0
    state_final = FillState(
        target=dummy_entity,
        state=FillLevel.FILLED,
        confidence=0.91,
        source="SigLIP2_Validated",
        timestamp=t_settled,
    )
    manager.update_state(entity_id, state_final)

    print(
        f" -> Final state confirmed: {state_final.state.value.upper()} (conf: {state_final.confidence:.2f}) at t=5.0s")
    print(f" -> Query IsEmpty():  {empty_pred()}")
    print(f" -> Query IsFilled(): {filled_pred()}")

    # -------------------------------------------------------------------------
    # Phase 5: Temporal Rollback Inspection
    # -------------------------------------------------------------------------
    print("\n[Phase 5: Temporal Rollback Query]")
    # Query at t0 + 0.5s (after Phase 1/2 perception, before Phase 3 pouring began at t0 + 2.0s)
    query_time = t0 + 0.5
    rollback_state = manager.get_state_at(entity_id, FillState, query_time)
    if rollback_state:
        print(f" -> State reconstructed at target t0+0.5s (pre-action): {rollback_state.state.value.upper()}")
    else:
        print(" -> [Error] No historical state matched timestamp.")

    print("\n" + "=" * 75)
    print("Demonstration successfully finished. All safety invariants preserved.")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
