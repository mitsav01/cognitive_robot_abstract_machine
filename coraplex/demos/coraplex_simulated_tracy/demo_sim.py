import os
import signal
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.alternative_motion_mappings.tracy_motion_mapping import (
    Arms,
    GripperState,
    TracyGripperMotion,
)
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer


def robotiq_gripper_dataclass_test():
    # 1. Forward virtual environment paths
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = ":".join(sys.path)

    # 2. Launch Giskard backend
    print("[INFO] Launching Giskard standalone backend...")
    giskard_process = subprocess.Popen(
        ["ros2", "launch", "giskardpy_ros", "giskardpy_tracy_standalone.launch.py"],
        start_new_session=True,
        env=child_env,
    )
    time.sleep(15)

    # 3. Initialize ROS 2 context and executor
    print("[INFO] Initializing ROS node and executor...")
    rclpy.init()
    node = rclpy.create_node("tracy_demo_node")
    ROSNodeRegistry().register(node)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin, daemon=True, name="rclpy-executor"
    )
    spin_thread.start()

    try:
        # 4. Fetch synchronized world
        print("[INFO] Fetching synchronized world...")
        world = fetch_world_from_service(node=node, timeout_seconds=300)
        WorldSynchronizer(_world=world, node=node)

        # 5. Connect GiskardWrapper
        giskard = GiskardWrapper(node_handle=node, world=world)

        # 6. Define gripper tasks
        task1 = TracyGripperMotion(gripper=Arms.BOTH, motion=GripperState.OPEN)
        task2 = TracyGripperMotion(gripper=Arms.BOTH, motion=GripperState.CLOSE)

        # 7. Execute Goal #0: Open Grippers
        print("[INFO] Executing Goal #0: Open Grippers...")
        msc_open = MotionStatechart()
        motion_chart_open = task1._motion_chart
        msc_open.add_node(motion_chart_open)
        msc_open.add_node(EndMotion.when_true(motion_chart_open))
        giskard.execute(msc_open)
        print("[INFO] Goal #0 (Open Grippers) completed.")

        # 8. Execute Goal #1: Close Grippers
        print("[INFO] Executing Goal #1: Close Grippers...")
        msc_close = MotionStatechart()
        motion_chart_close = task2._motion_chart
        msc_close.add_node(motion_chart_close)
        msc_close.add_node(EndMotion.when_true(motion_chart_close))
        giskard.execute(msc_close)
        print("[INFO] Goal #1 (Close Grippers) completed.")

    finally:
        # 9. Clean shutdown
        print("[INFO] Cleaning up processes and shutting down ROS...")
        executor.shutdown()
        ROSNodeRegistry().clear(node)
        node.destroy_node()
        rclpy.shutdown()

        os.killpg(os.getpgid(giskard_process.pid), signal.SIGTERM)
        giskard_process.wait()


if __name__ == "__main__":
    robotiq_gripper_dataclass_test()