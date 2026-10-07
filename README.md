# VESPER: Action-Aware Multimodal Object State Verification for Semantic Digital Twins

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_%7C_Jazzy-22314E?logo=ros&logoColor=white)](https://docs.ros.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/Tests-16%20Passing-brightgreen)](https://github.com/)

**VESPER** is an action-aware temporal perception and state-verification framework designed for robotic manipulation. It
combines open-vocabulary visual perception, temporal evidence arbitration, robot action context, and Semantic Digital
Twins (SDT) to convert uncertain visual observations into persistent, symbolic object-state representations.

---

## Table of Contents

- [Overview](#overview)
- [Key Idea & Conceptual Pipeline](#key-idea--conceptual-pipeline)
- [Core Contributions](#core-contributions)
- [System Architecture](#system-architecture)
- [Multimodal Perception Pipeline](#multimodal-perception-pipeline)
- [State Representation & Transitions](#state-representation--transitions)
- [Repository Structure](#repository-structure)
- [Hardware & Software Stack](#hardware--software-stack)
- [Dataset & Evaluation](#dataset--evaluation)
  - [Dataset Composition](#dataset-composition)
  - [Experimental Results](#experimental-results)
- [Quickstart & Verification Demo](#quickstart--verification-demo)
- [Running Automated Tests](#running-automated-tests)
- [Design Principles & Research Questions](#design-principles--research-questions)
- [Citation](#citation)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview

Robots operating in open-world environments must continuously verify whether objects have changed physical state as a
consequence of manipulation actions:

- *Is a container still empty?*
- *Has a vessel become filled after a pouring trajectory?*
- *Has an ingredient been cut?*
- *Did an observed transition actually occur, or was it a transient visual artifact caused by gripper occlusion or
  motion blur?*

Modern Vision-Language Models (VLMs) offer rich open-vocabulary visual reasoning, but **a frame-level inference should
not automatically become a persistent symbolic belief**.

Visual evidence during manipulation is degraded by:

1. **Perceptual variability:** Fluctuations in illumination, viewpoint, sensor noise, and specular reflections.
2. **Action-induced occlusion:** Physical obstruction of the object by the robot arm, end-effector, or manipulated
   tools.
3. **Temporal prediction instability:** Frame-to-frame label flicker inherent to zero-shot neural classifiers.
4. **Delayed physical observables:** Physical state changes whose final visual evidence only manifests once an action
   completes and clearance is restored.
5. **The Neural-Symbolic semantic gap:** Unreconciled perceptual data prematurely corrupting high-level planning
   representations.

VESPER bridges this gap via an intermediate **action-aware temporal arbitration layer** situated between continuous
visual evidence and symbolic state commitment.

> **Central Principle:**  
> *A visual observation is evidence about an object state; it is not automatically the committed symbolic state of the
object.*

---

## Key Idea & Conceptual Pipeline

VESPER decouples transient perception from persistent belief updates across five explicit stages:

$$\text{Observation} \longrightarrow \text{Evidence} \longrightarrow \text{Arbitration} \longrightarrow \text{Commitment} \longrightarrow \text{Representation}$$

```
        RGB-D Camera + Robot Action Context
                         │
                         ▼
              Object / Region Grounding
                    OWLv2 / Vision
                         │
                         ▼
                VLM State Evidence
                  SigLIP 2 / VLM
                         │
                         ▼
              Temporal State Arbitration
                         │
             ┌───────────┴───────────┐
             │                       │
       Evidence Unstable       Evidence Supported
             │                       │
             ▼                       ▼
       Hold / Reject            Commit State
                                     │
                                     ▼
                           Semantic Digital Twin
                                     │
                                     ▼
                         Persistent Symbolic State
```

---

## Core Contributions

### 1. Action-Aware Temporal State Arbitration

VESPER factors execution phase context into perceptual interpretation. During dynamic manipulation (e.g., pouring), the
manipulator frequently occludes the Region of Interest (RoI). Instead of reacting to momentary occlusion artifacts,
VESPER flags observations during manipulation as **transitional evidence**, deferring symbolic commitment until
post-action visual clearance is restored.

```
Action Starts ──► Visual Access Drops ──► Hold Candidate ──► Action Ends ──► View Restored ──► Commit to SDT
```

### 2. Bounded Temporal State Memory

The `DynamicStateManager` maintains historical object-state observations through time-bounded ring buffers
(`deque(maxlen=N)`):

* Object-indexed temporal state tracking
* Timestamped observation logs with confidence metrics and provenance
* Duplicate observation suppression and sliding-window pruning
* Point-in-time state reconstruction: `get_state_at(timestamp)`

### 3. Evidence-Aware State Commitment

Belief stability does not equal physical correctness. Repeated noisy classifications can appear stable without being
physically true. VESPER decouples:

- **Visual Evidence:** Instantaneous VLM similarity scores.
- **Candidate State:** Uncommitted target label hypotheses.
- **Temporal Support:** Rolling multi-frame observation density.
- **Action Context:** Execution status of the physical robot.
- **Committed State:** Grounded symbolic predicate in the digital twin.

### 4. Semantic Digital Twin (SDT) Integration

Verified states mutate the underlying Semantic Digital Twin, exposing typed symbolic predicates for task planning and
plan monitoring:

* `IsEmpty(object)`
* `IsFilled(object)`
* `IsFull(object)`
* `IsCut(object)`

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Physical Robot System                    │
│                                                             │
│  RGB-D Camera                          Robot Actions        │
│      │                                      │               │
└──────┼──────────────────────────────────────┼───────────────┘
       │                                      │
       ▼                                      ▼
┌──────────────────┐                 ┌──────────────────┐
│ Object Grounding │                 │  Action Context  │
│     OWLv2        │                 │  Pour / Fill /   │
│                  │                 │   Manipulation   │
└────────┬─────────┘                 └────────┬─────────┘
         │                                    │
         └────────────────┬───────────────────┘
                          ▼
                 ┌───────────────────┐
                 │   VLM Evidence    │
                 │     SigLIP 2      │
                 │ state + confidence│
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Temporal + Action │
                 │    Arbitration    │
                 │ hold / reject     │
                 │ support / commit  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Dynamic State   │
                 │      Manager      │
                 │  bounded history  │
                 │  provenance cache │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Semantic Digital  │
                 │       Twin        │
                 │  symbolic state   │
                 │    predicates     │
                 └───────────────────┘
```

---

## Multimodal Perception Pipeline

The runtime pipeline executes across seven modular stages:

```
Observe ──► Estimate ──► Stabilize ──► Verify ──► Validate ──► Commit ──► Represent
```

1. **Observe:** Ingest calibrated RGB-D frames from the active camera feed.
2. **Estimate:** Ground target objects via **OWLv2** bounding box proposals; query **SigLIP 2** for zero-shot
   text-prompt similarity against state hypotheses (e.g., `"empty mug"`, `"filled mug"`, `"full mug"`).
3. **Stabilize:** Buffer observations and suppress consecutive duplicate reports.
4. **Verify:** Screen candidate states against temporal confidence thresholds.
5. **Validate:** Cross-reference active robot trajectories; defer commitment during unverified or occluded motions.
6. **Commit:** Mutate persistent state within `DynamicStateManager` once arbitration criteria are satisfied.
7. **Represent:** Propagate validated predicates into the Semantic Digital Twin for downstream executive systems (e.g.,
   CRAM).

---

## State Representation & Transitions

The primary experimental prototype handles container manipulation states:

```
                 ┌─────────────┐
                 │    EMPTY    │
                 └──────┬──────┘
                        │ Fill / Pour
                        ▼
                 ┌─────────────┐
                 │   FILLED    │
                 └──────┬──────┘
                        │ Continued Fill
                        ▼
                 ┌─────────────┐
                 │    FULL     │
                 └─────────────┘
```

Observation-conditioned transitions accommodate real-world perceptual uncertainty:

* $\text{EMPTY} \longrightarrow \{\text{EMPTY}, \text{FILLED}\}$
* $\text{FILLED} \longrightarrow \{\text{EMPTY}, \text{FILLED}, \text{FULL}\}$
* $\text{FULL} \longrightarrow \{\text{FILLED}, \text{FULL}\}$

*(Extensions for cutting and food preparation support states such as `CUT` and `UNCUT`.)*

---

## Repository Structure

| Component                    | Filepath                                                                                | Description                                           |
|:-----------------------------|:----------------------------------------------------------------------------------------|:------------------------------------------------------|
| **Object State Abstraction** | `semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/object_state.py`  | Core semantic state definitions and classes           |
| **Dynamic State Manager**    | `semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/state_manager.py` | Bounded temporal state memory and arbitration engine  |
| **EQL Predicates**           | `semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/object_state.py`  | Symbolic state predicates for digital twin queries    |
| **Object-State Tests**       | `test/semantic_digital_twin_test/test_semantic_annotations/test_object_state.py`        | Unit tests for state models and mutations             |
| **State-Manager Tests**      | `test/semantic_digital_twin_test/test_semantic_annotations/test_state_manager.py`       | Unit tests for temporal buffers and arbitration logic |
| **Verification Demo**        | `experiments/src/experiments/object_state_estimation/run_object_state_verification.py`  | Standalone demonstration pipeline                     |
| **Action Execution**         | `coraplex/`                                                                             | Robot skill dispatching and action state tracking     |
| **Motion Planning**          | `giskardpy/`                                                                            | Whole-body motion planning and kinematic execution    |

---

## Hardware & Software Stack

* **Manipulator Platform:** PAL Robotics TIAGo
* **Perception Hardware:** RGB-D Head Camera (structure-light / ToF)
* **Vision Backends:** OWLv2 (open-vocabulary grounding), SigLIP 2 (vision-language similarity), OpenCV
* **Middleware & Environment:** ROS 2 (Humble / Jazzy), PyTorch
* **Cognitive Architecture:** CRAM, GiskardPy, Semantic Digital Twin (SDT)

---

## Dataset & Evaluation

### Dataset Composition

The benchmark evaluation comprises tabletop manipulation observations collected across multiple experimental kitchen
environments (`vorstrasse`, `iai_kueche`), alongside **25 continuous pouring interaction videos**.

| Class Label      | Frame Count |
|:-----------------|:------------|
| `EMPTY`          | 347         |
| `FILLED`         | 209         |
| `FULL`           | 161         |
| **Total Frames** | **717**     |

### Experimental Results

| Metric                       | Baseline VLM (Per-Frame) | VESPER (Canonical Temporal Arbitration) |
|:-----------------------------|:------------------------:|:---------------------------------------:|
| **Frame Accuracy**           |        **81.73%**        |                 66.14%*                 |
| **Macro-F1**                 |        **80.45%**        |                 58.07%                  |
| **State-Switch Rate**        |            —             |              **4.146 s⁻¹**              |
| **Settlement Rate**          |            —             |            **56.2%** (9/16)             |
| **Final-State Verification** |            —             |            **72.0%** (18/25)            |
| **Action Support Ratio**     |            —             |                **93.8%**                |
| **Mean Evidence Support**    |            —             |                **0.504**                |

> *\*Note on Frame-Level Metrics:*  
> Frame accuracy penalizes deferred commits during active transitions because the ground-truth label updates before
> visual clearance allows safe arbitration. VESPER trades instantaneous frame-matching accuracy for symbolic belief
> stability, successfully suppressing transient state-flicker during physical manipulation.

### Computational Latency

* **Perception (OWLv2 + SigLIP 2):** $\sim 150 - 300\text{ ms / frame}$
* **Arbitration Engine:** $< 0.5\text{ ms / frame}$
* **Arbitration Overhead:** $\sim 0.34\%$

---

## Quickstart & Verification Demo

To test temporal state management, duplicate suppression, and action-context arbitration independently of the full robot
hardware stack:

```bash
# Clone the repository
git clone https://github.com/your-org/vesper.git
cd vesper

# Run the standalone verification demonstration
python experiments/src/experiments/object_state_estimation/run_object_state_verification.py
```

### Trace Output

```text
============================================================
VESPER: Dynamic State Verification & Temporal Memory Demo
============================================================

[Phase 1: Initial Perception]
Observation:     EMPTY (conf: 0.94)
Semantic query:  IsEmpty() -> True | IsFilled() -> False

[Phase 2: Repeated Observation]
Repeated EMPTY evidence received.
Temporal manager: duplicate observation suppressed.

[Phase 3: Active Manipulation]
Robot action:    POURING
Visual access:   partially occluded
Candidate:       FILLED (conf: 0.45)
Arbitration:     transitional evidence -> commitment deferred

[Phase 4: Post-Action Settlement]
Robot action:    COMPLETED
Visual access:   restored
Candidate:       FILLED (conf: 0.91)
Arbitration:     support criteria met -> state committed to FILLED

[Phase 5: Temporal Query]
Historical query: get_state_at(t_pre_action)
Result:           EMPTY
============================================================
Verification demonstration completed
============================================================
```

---

## Running Automated Tests

The framework includes a unit-test suite checking temporal container bounds, timestamp preservation, and predicate
integrity:

```bash
# Test semantic state model abstractions
pytest test/semantic_digital_twin_test/test_semantic_annotations/test_object_state.py -v

# Test dynamic state manager and ring-buffer arbitration
pytest test/semantic_digital_twin_test/test_semantic_annotations/test_state_manager.py -v
```

---

## Design Principles & Research Questions

1. **Observation $\neq$ Belief:** Sensory detection is inconclusive evidence, not ground-truth reality.
2. **Stability $\neq$ Correctness:** Persistent erroneous classifications must be checked against action plausibility.
3. **Action Context is Foundational:** The physical state change is intrinsically coupled to the robot's physical
   trajectory.
4. **Symbolic Persistence:** World states must remain invariant to momentary sensory dropouts or sensor occlusions.

### Research Questions

* **RQ1 (Temporal Robustness):** Can temporal arbitration suppress spurious state transitions caused by stochastic
  frame-level VLM outputs?
* **RQ2 (Action Awareness):** Does incorporating robot execution context resolve sensory ambiguities induced by physical
  occlusion?
* **RQ3 (Symbolic Commitment):** How can probabilistic neural similarity scores be reliably mapped onto discrete,
  persistent world models?
* **RQ4 (Digital Twin Grounding):** Can verified perceptual hypotheses update a Semantic Digital Twin sufficiently to
  support closed-loop task execution?

---

## Citation

If you use VESPER in your academic research, please cite the following work:

```bibtex
@mastersthesis{savsaviya2026vesper,
  author  = {Savsaviya, Mitesh},
  title   = {Action-Aware Neuro-Symbolic Object State Verification using Vision-Language Models and Semantic Digital Twins},
  school  = {University of Bremen},
  year    = {2026}
}
```

---

## License & Acknowledgements

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

This research was developed within the **Cognitive Systems / Robotics Group at the University of Bremen (AICOR)**,
integrating with the **CRAM** and **Semantic Digital Twin** ecosystem.