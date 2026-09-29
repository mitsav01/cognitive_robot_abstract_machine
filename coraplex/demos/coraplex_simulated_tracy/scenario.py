import logging
import os
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Execution mode: SIMULATED (standalone Giskard/bullet) or REAL (hardware)
EXECUTION_MODE = ExecutionType.SIMULATED


def main():
    # 1. Initialize ROS 2 context, node, and background spinning
    print("[INFO] Initializing ROS 2 context...")
    rclpy.init()
    node = rclpy.create_node("tracy_coraplex_demo_node")

    # Register node with Giskard ROSNodeRegistry and internal rospy wrapper
    ROSNodeRegistry().register(node)
    rospy.init_node(node.get_name())

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin, daemon=True, name="rclpy-executor"
    )
    spin_thread.start()

    try:
        # 2. Fetch the canonical synchronized world state from tracy_standalone
        print("[INFO] Fetching synchronized world from service...")
        world = fetch_world_from_service(node=node, timeout_seconds=120)

        # 3. Attach synchronizer immediately so delta updates are tracked
        world_sync = WorldSynchronizer(_world=world, node=node)

        # 4. Pass the identical synchronized world instance to GiskardWrapper
        print("[INFO] Connecting to Giskard server...")
        giskard = GiskardWrapper(node_handle=node, world=world)

        # 5. Extract semantic Tracy robot description
        tracy_annotations = world.get_semantic_annotations_by_type(Tracy)
        if not tracy_annotations:
            raise RuntimeError("Tracy robot description not found in world state.")
        robot_view = tracy_annotations[0]

        # 6. Check if cup is already in the synchronized world; merge only if absent
        cup_name = "jeroen_cup"
        existing_cups = [b for b in world.bodies if cup_name in str(b.name)]

        if not existing_cups:
            resources_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "resources", "objects"
            )
            cup_path = os.path.join(resources_dir, "jeroen_cup.stl")
            if not os.path.exists(cup_path):
                cup_path = os.path.abspath(
                    os.path.join(
                        os.path.expanduser("~"),
                        "cram_ws/cognitive_robot_abstract_machine/coraplex/resources/objects/jeroen_cup.stl",
                    )
                )

            cup = STLParser(cup_path).parse()
            cup_body = cup.root

            # Ensure collision shapes are populated for bounding box resolution
            if not cup_body.collision or len(cup_body.collision) == 0:
                cup_body.collision = cup_body.visual

            with world.modify_world():
                world.merge_world(
                    cup,
                    FixedConnection(
                        world.root,
                        cup_body,
                        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_quaternion(
                            0.5, 0.15, 0.966, reference_frame=world.root
                        ),
                    ),
                )
                cup_designator = HasRootBody(root=cup_body)
                world.add_semantic_annotations([cup_designator])
        else:
            cup_body = existing_cups[0]
            annotations = world.get_semantic_annotations_by_type(HasRootBody)
            cup_designator = annotations[0] if annotations else HasRootBody(root=cup_body)

        print(f"[INFO] Target cup body: {cup_body.name}")
        print(f"[INFO] World Root: {world.root.name}")

        # 7. Build planning Context
        context = Context(world, robot_view, ros_node=node)

        # 8. Define Grasp Specification
        end_effector = (
            context.robot.get_left_arm_if_specified().end_effector
            if hasattr(context.robot, "get_left_arm_if_specified")
            else context.robot.left_arm.end_effector
        )

        pick_up_grasp = GraspDescription(
            approach_direction=ApproachDirection.LEFT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            end_effector=end_effector,
            manipulation_offset=0.02,
        )

        # 9. Assemble Sequential Pick-and-Place Plan
        plan = sequential(
            [
                ParkArmsAction(arm=Arms.BOTH),
                SetGripperAction(gripper=Arms.BOTH, motion=GripperState.OPEN),
                PickUpAction(cup_designator, Arms.LEFT, pick_up_grasp),
                PlaceAction(
                    cup_body,
                    HomogeneousTransformationMatrix.from_xyz_rpy(
                        0.6, -0.1, 0.966, reference_frame=world.root
                    ).to_pose(),
                    Arms.LEFT,
                ),
                ParkArmsAction(arm=Arms.BOTH),
            ],
            context,
        )

        # 10. Execute Plan using ExecutionEnvironment
        print(f"[INFO] Performing plan with ExecutionEnvironment ({EXECUTION_MODE.name})...")
        with ExecutionEnvironment(execution_type=EXECUTION_MODE, collision_avoidance=False):
            plan.perform()

        print("[INFO] Plan completed successfully.")

    finally:
        # 11. Clean shutdown of executor and ROS node
        print("[INFO] Shutting down executor and node...")
        executor.shutdown()
        ROSNodeRegistry().clear(node)
        node.destroy_node()
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
