# BSD 3-Clause License
#
# Copyright (c) 2022 InOrbit, Inc.
# Copyright (c) 2022 Clearpath Robotics, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the InOrbit, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rclpy
import pytest
import threading
import time
from rclpy.logging import LoggingSeverity
from rclpy.task import Future
from rclpy.parameter import Parameter
from action_msgs.msg import GoalStatus

from uuid import uuid4
from unittest.mock import MagicMock
from types import SimpleNamespace

from vda5050_connector_py.vda5050_controller import VDA5050Controller
from vda5050_connector_py.vda5050_controller import OrderAcceptModes
from vda5050_connector_py.vda5050_controller import OrderRejectErrors
from vda5050_connector_py.utils import get_vda5050_ts
from vda5050_connector.action import NavigateToNode
from vda5050_connector.action import NavigateThroughNodes
from vda5050_connector.srv import ExtendNavigation
from vda5050_msgs.msg import Order
from vda5050_msgs.msg import Node
from vda5050_msgs.msg import Edge
from vda5050_msgs.msg import NodePosition
from vda5050_msgs.msg import Action
from vda5050_msgs.msg import ActionParameter
from vda5050_msgs.msg import InstantActions
from vda5050_msgs.msg import CurrentAction
from vda5050_msgs.msg import Error
from vda5050_msgs.msg import OrderState


def succeeded_action_result_future(action_type):
    """Build the GetResult response returned by a successful ROS action goal."""
    return action_result_future(action_type, GoalStatus.STATUS_SUCCEEDED)


def action_result_future(action_type, status):
    """Build a ROS action result future with the requested terminal status."""
    response = action_type.Impl.GetResultService.Response()
    response.status = status
    future = Future()
    future.set_result(result=response)
    return future


def get_order_new(order_id=str(uuid4()), order_update_id=0):
    return Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=order_update_id,
        nodes=[
            Node(
                node_id="node1",
                sequence_id=0,
                released=True,
                node_position=NodePosition(
                    x=2.0,
                    y=0.95,
                    theta=-0.66,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node2",
                sequence_id=2,
                released=True,
                node_position=NodePosition(
                    x=1.18,
                    y=-1.76,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node3",
                sequence_id=4,
                released=True,
                node_position=NodePosition(
                    x=-0.38,
                    y=1.89,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node4",
                sequence_id=6,
                released=True,
                node_position=NodePosition(
                    x=-0.17,
                    y=1.74,
                    theta=-2.6,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node1",
                sequence_id=8,
                released=True,
                node_position=NodePosition(
                    x=2.0,
                    y=0.95,
                    theta=-0.66,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
        ],
        edges=[
            Edge(
                edge_id="edge1",
                sequence_id=1,
                released=True,
                start_node_id="node1",
                end_node_id="node2",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
            Edge(
                edge_id="edge2",
                sequence_id=3,
                released=True,
                start_node_id="node2",
                end_node_id="node3",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
            Edge(
                edge_id="edge3",
                sequence_id=5,
                released=True,
                start_node_id="node3",
                end_node_id="node4",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
            Edge(
                edge_id="edge4",
                sequence_id=7,
                released=True,
                start_node_id="node4",
                end_node_id="node1",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
        ],
    )


def test_order_msg_is_valid_accepts_standard_graph():
    order = get_order_new()

    is_valid, error = VDA5050Controller.order_msg_is_valid(None, order)

    assert is_valid is True
    assert error.error_description == ""


@pytest.mark.parametrize("active_navigation", [
    {"_nav_through_nodes_goal_pending": True},
    {"_navigate_through_nodes_goal_handle": object()},
    {"_navigate_to_node_goal_handle": object()},
    {"_pending_nav_through_nodes_result": object()},
])
def test_navigation_action_alone_does_not_block_new_order(active_navigation):
    controller = SimpleNamespace(
        _canceling_order=lambda: False,
        _current_state=OrderState(),
        _instant_action_ids=set(),
        **active_navigation,
    )

    assert not VDA5050Controller._has_current_order(controller)


@pytest.mark.parametrize(
    ("goal_order_generation", "has_accepted_goal", "expected"),
    [
        (5, True, True),
        (4, True, False),
        (5, False, False),
    ],
)
def test_only_accepted_current_order_goal_can_be_extended(
    goal_order_generation, has_accepted_goal, expected
):
    controller = SimpleNamespace(
        _order_generation=5,
        _nav_through_nodes_goal_order_generation=goal_order_generation,
        _navigate_through_nodes_goal_handle=(object() if has_accepted_goal else None),
    )

    assert (
        VDA5050Controller._is_current_order_navigation_goal(controller)
        is expected
    )


@pytest.mark.parametrize(
    "status", [GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED]
)
def test_stale_navigation_result_does_not_advance_replacement_order(status):
    controller = SimpleNamespace(
        _order_generation=2,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_order_generation=1,
        _navigate_through_nodes_goal_handle=object(),
        _canceling_order=lambda: False,
        _on_active_order=MagicMock(),
        _process_last_edge_node=MagicMock(),
        _handle_navigation_failure=MagicMock(),
        _nav_through_nodes_extensions_pending=0,
        logger=MagicMock(),
    )
    future = action_result_future(NavigateThroughNodes, status)

    VDA5050Controller._navigate_through_nodes_result_callback(
        controller, future, order_generation=1
    )

    assert controller._navigate_through_nodes_goal_handle is None
    controller._process_last_edge_node.assert_not_called()
    controller._handle_navigation_failure.assert_not_called()
    controller._on_active_order.assert_called_once_with()


def test_stale_navigation_feedback_does_not_advance_replacement_order():
    controller = SimpleNamespace(
        _order_generation=2,
        _current_node_goal=SimpleNamespace(sequence_id=4),
        _canceling_order=lambda: False,
        _process_last_edge_node=MagicMock(),
    )
    feedback = SimpleNamespace(
        feedback=SimpleNamespace(
            last_node=SimpleNamespace(sequence_id=4)
        )
    )

    VDA5050Controller._navigate_through_nodes_feedback_callback(
        controller, feedback, order_generation=1
    )

    controller._process_last_edge_node.assert_not_called()
    assert controller._current_node_goal.sequence_id == 4


def test_stale_navigation_result_waits_for_pending_stitch_response():
    controller = SimpleNamespace(
        _order_generation=2,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_order_generation=1,
        _navigate_through_nodes_goal_handle=object(),
        _canceling_order=lambda: False,
        _on_active_order=MagicMock(),
        _process_last_edge_node=MagicMock(),
        _handle_navigation_failure=MagicMock(),
        _nav_through_nodes_extensions_pending=1,
        _pending_nav_through_nodes_result=None,
        _pending_nav_through_nodes_generation=None,
        logger=MagicMock(),
    )
    controller._navigate_through_nodes_result_callback = (
        lambda future, order_generation=None:
        VDA5050Controller._navigate_through_nodes_result_callback(
            controller, future, order_generation=order_generation
        )
    )
    future = succeeded_action_result_future(NavigateThroughNodes)

    VDA5050Controller._navigate_through_nodes_result_callback(
        controller, future, order_generation=1
    )

    assert controller._pending_nav_through_nodes_result is future
    assert controller._pending_nav_through_nodes_generation == 1
    controller._on_active_order.assert_not_called()

    controller._nav_through_nodes_extensions_pending = 0
    VDA5050Controller._resume_pending_navigation_result(controller)

    assert controller._pending_nav_through_nodes_result is None
    controller._process_last_edge_node.assert_not_called()
    controller._on_active_order.assert_called_once_with()


def test_rejected_stale_navigation_goal_reschedules_current_order():
    controller = SimpleNamespace(
        _order_generation=2,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_order_generation=1,
        _nav_through_nodes_goal_pending=True,
        _navigate_through_nodes_goal_handle=object(),
        _current_node_goal=object(),
        _on_active_order=MagicMock(),
        logger=MagicMock(),
    )
    future = Future()
    future.set_result(result=SimpleNamespace(accepted=False))

    VDA5050Controller._navigate_through_nodes_goal_response_callback(
        controller, future, order_generation=1
    )

    assert controller._nav_through_nodes_goal_pending is False
    assert controller._navigate_through_nodes_goal_handle is None
    controller._on_active_order.assert_called_once_with()


def test_current_navigation_failure_is_still_reported():
    controller = SimpleNamespace(
        _order_generation=3,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_order_generation=3,
        _navigate_through_nodes_goal_handle=object(),
        _canceling_order=lambda: False,
        _handle_navigation_failure=MagicMock(),
        _on_active_order=MagicMock(),
    )

    VDA5050Controller._navigate_through_nodes_result_callback(
        controller,
        action_result_future(NavigateThroughNodes, GoalStatus.STATUS_ABORTED),
        order_generation=3,
    )

    controller._handle_navigation_failure.assert_called_once_with()
    controller._on_active_order.assert_not_called()


def test_failed_navigation_extension_is_reported_as_information_only():
    existing_error = Error(error_type="NAVIGATION_ERROR")
    controller = SimpleNamespace(
        _nav_through_nodes_extensions_pending=1,
        _current_state=OrderState(
            errors=[existing_error],
            informations=[],
        ),
        _resume_pending_navigation_result=MagicMock(),
        logger=MagicMock(),
    )

    def update_state(partial_state, publish_now=False):
        for field, value in partial_state.items():
            setattr(controller._current_state, field, value)

    controller._update_state = MagicMock(side_effect=update_state)
    future = SimpleNamespace(result=lambda: SimpleNamespace(
        success=False,
        message="AGV is in a segment switch stop transition.",
    ))

    VDA5050Controller._extend_navigation_response_callback(
        controller, last_seq=4, future=future
    )

    assert controller._current_state.errors == [existing_error]
    assert len(controller._current_state.informations) == 1
    info = controller._current_state.informations[0]
    assert info.info_type == "NAVIGATION_EXTENSION_DEFERRED"
    assert "segment switch stop transition" in info.info_description
    assert info.info_level == info.INFO
    controller._resume_pending_navigation_result.assert_called_once_with()


def test_navigation_callbacks_capture_order_generation():
    action_client = MagicMock()
    controller = SimpleNamespace(
        _navigate_through_nodes_act_cli=action_client,
        _order_generation=7,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_pending=False,
        _nav_through_nodes_goal_order_generation=None,
        _navigate_through_nodes_goal_handle=None,
        _current_node_goal=None,
        _nav_through_nodes_last_seq=0,
        _navigate_through_nodes_feedback_callback=MagicMock(),
        _navigate_through_nodes_goal_response_callback=MagicMock(),
        logger=MagicMock(),
    )
    send_goal_future = SimpleNamespace(add_done_callback=MagicMock())
    action_client.send_goal_async.return_value = send_goal_future

    VDA5050Controller.send_adapter_navigate_through_nodes(
        controller,
        edges=[Edge(edge_id="edge-1", sequence_id=1)],
        nodes=[Node(node_id="P2", sequence_id=2)],
    )

    feedback_callback = action_client.send_goal_async.call_args.kwargs[
        "feedback_callback"
    ]
    response_callback = send_goal_future.add_done_callback.call_args.args[0]
    assert feedback_callback.keywords["order_generation"] == 7
    assert response_callback.keywords["order_generation"] == 7


def test_duplicate_navigation_dispatch_is_reserved_before_waiting_for_server():
    controller = SimpleNamespace(
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_pending=False,
        _nav_through_nodes_goal_order_generation=None,
        _navigate_through_nodes_goal_handle=None,
        _order_generation=3,
        _current_node_goal=None,
        _nav_through_nodes_last_seq=0,
        _navigate_through_nodes_feedback_callback=MagicMock(),
        _navigate_through_nodes_goal_response_callback=MagicMock(),
        logger=MagicMock(),
    )
    action_client = MagicMock()
    controller._navigate_through_nodes_act_cli = action_client
    send_goal_future = SimpleNamespace(add_done_callback=MagicMock())
    action_client.send_goal_async.return_value = send_goal_future
    edges = [Edge(edge_id="edge-1", sequence_id=1)]
    nodes = [Node(node_id="P2", sequence_id=2)]

    def dispatch_again_while_waiting():
        VDA5050Controller.send_adapter_navigate_through_nodes(
            controller, edges=edges, nodes=nodes
        )
        return True

    action_client.wait_for_server.side_effect = dispatch_again_while_waiting

    VDA5050Controller.send_adapter_navigate_through_nodes(
        controller, edges=edges, nodes=nodes
    )

    action_client.send_goal_async.assert_called_once()
    assert controller._nav_through_nodes_goal_pending is True


def test_late_rejected_goal_response_does_not_clear_accepted_goal():
    result_future = SimpleNamespace(add_done_callback=MagicMock())
    accepted_handle = SimpleNamespace(
        accepted=True,
        get_result_async=MagicMock(return_value=result_future),
    )
    controller = SimpleNamespace(
        _order_generation=3,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_pending=True,
        _nav_through_nodes_goal_order_generation=3,
        _navigate_through_nodes_goal_handle=None,
        _navigate_through_nodes_result_callback=MagicMock(),
        _canceling_order=lambda: False,
        logger=MagicMock(),
    )

    accepted_future = Future()
    accepted_future.set_result(result=accepted_handle)
    VDA5050Controller._navigate_through_nodes_goal_response_callback(
        controller, accepted_future, order_generation=3
    )

    assert controller._nav_through_nodes_goal_pending is False
    assert controller._navigate_through_nodes_goal_handle is accepted_handle
    result_future.add_done_callback.assert_called_once()

    late_rejected_future = Future()
    late_rejected_future.set_result(result=SimpleNamespace(accepted=False))
    VDA5050Controller._navigate_through_nodes_goal_response_callback(
        controller, late_rejected_future, order_generation=3
    )

    assert controller._navigate_through_nodes_goal_handle is accepted_handle
    accepted_handle.get_result_async.assert_called_once()


def test_cancel_order_waits_for_pending_navigation_acceptance_once():
    cancel_action_state = CurrentAction(
        action_id="cancel-1", action_status=CurrentAction.WAITING)
    waiting_action_state = CurrentAction(
        action_id="order-action", action_status=CurrentAction.WAITING)
    controller = SimpleNamespace(
        _cancel_action=Action(
            action_id="cancel-1", action_type="cancelOrder"),
        _cancel_clear_error_pending=False,
        _current_state=OrderState(
            action_states=[cancel_action_state, waiting_action_state]),
        _current_order=Order(order_id="order-1"),
        _process_vda_action_goal_handle_dict={},
        _enable_nav_through_nodes=True,
        _navigate_through_nodes_goal_handle=None,
        logger=MagicMock(),
    )

    def get_action_status(action_id):
        return next(
            state.action_status for state in controller._current_state.action_states
            if state.action_id == action_id
        )

    def update_action_status(action_id, status):
        action_state = next(
            state for state in controller._current_state.action_states
            if state.action_id == action_id
        )
        action_state.action_status = status

    controller._get_action_status = get_action_status
    controller._update_action_status = MagicMock(side_effect=update_action_status)
    controller._has_current_order = MagicMock(return_value=True)
    controller._is_navigation_active = MagicMock(return_value=True)

    VDA5050Controller._cancel_order(controller)
    VDA5050Controller._cancel_order(controller)

    assert cancel_action_state.action_status == CurrentAction.RUNNING
    assert waiting_action_state.action_status == CurrentAction.FAILED
    controller._update_action_status.assert_any_call(
        "cancel-1", CurrentAction.RUNNING)
    controller._update_action_status.assert_any_call(
        "order-action", CurrentAction.FAILED)
    assert controller._update_action_status.call_count == 2
    assert controller.logger.info.call_count == 1


def test_cancel_order_sends_navigation_cancel_once():
    cancel_goal = MagicMock()
    cancel_action_state = CurrentAction(
        action_id="cancel-2", action_status=CurrentAction.WAITING)
    controller = SimpleNamespace(
        _cancel_action=Action(
            action_id="cancel-2", action_type="cancelOrder"),
        _cancel_clear_error_pending=False,
        _current_state=OrderState(action_states=[cancel_action_state]),
        _current_order=Order(order_id="order-2"),
        _process_vda_action_goal_handle_dict={},
        _enable_nav_through_nodes=True,
        _navigate_through_nodes_goal_handle=SimpleNamespace(
            cancel_goal_async=cancel_goal),
        logger=MagicMock(),
    )
    controller._get_action_status = lambda action_id: cancel_action_state.action_status

    def update_action_status(action_id, status):
        cancel_action_state.action_status = status

    controller._update_action_status = MagicMock(side_effect=update_action_status)
    controller._has_current_order = MagicMock(return_value=True)
    controller._is_navigation_active = MagicMock(return_value=True)

    VDA5050Controller._cancel_order(controller)
    VDA5050Controller._cancel_order(controller)

    cancel_goal.assert_called_once_with()


def test_only_one_cancel_order_is_handled_per_order_generation():
    controller = SimpleNamespace(
        _cancel_action=None,
        _cancel_attempted_generation=None,
        _order_generation=4,
        _instant_action_ids=set(),
        _current_state=OrderState(action_states=[]),
        logger=MagicMock(),
        _publish_state=MagicMock(),
        instant_action_msg_is_valid=lambda actions: (True, None),
    )

    def update_state(partial_state):
        for field, value in partial_state.items():
            setattr(controller._current_state, field, value)

    def update_action_status(action_id, status, result_description=""):
        action_state = next(
            action for action in controller._current_state.action_states
            if action.action_id == action_id
        )
        action_state.action_status = status
        if result_description:
            action_state.result_description = result_description

    controller._update_state = update_state
    controller._update_action_status = update_action_status
    cancel_actions = InstantActions(actions=[
        Action(action_id="cancel-a", action_type="cancelOrder",
               blocking_type=Action.NONE),
        Action(action_id="cancel-b", action_type="cancelOrder",
               blocking_type=Action.NONE),
    ])

    VDA5050Controller.process_instant_actions(controller, cancel_actions)

    action_states = {
        action.action_id: action for action in controller._current_state.action_states
    }
    assert controller._cancel_action.action_id == "cancel-a"
    assert controller._cancel_attempted_generation == 4
    assert action_states["cancel-a"].action_status == CurrentAction.WAITING
    assert action_states["cancel-b"].action_status == CurrentAction.FAILED
    assert "already handled" in action_states["cancel-b"].result_description
    controller._publish_state.assert_called_once_with()


def test_cancel_order_waits_for_adapter_confirmation():
    cancel_action = Action(action_id="cancel-wait", action_type="cancelOrder")
    controller = SimpleNamespace(
        _cancel_action=cancel_action,
        _cancel_clear_error_pending=False,
        _get_action_status=MagicMock(return_value=CurrentAction.RUNNING),
        _update_action_status=MagicMock(),
        _publish_state=MagicMock(),
        logger=MagicMock(),
        _process_vda_action_goal_handle_dict={"still-running": object()},
        _is_navigation_active=MagicMock(return_value=True),
    )

    VDA5050Controller._cancel_order(controller)

    assert controller._cancel_action is cancel_action
    controller._update_action_status.assert_not_called()
    controller._publish_state.assert_not_called()
    controller._is_navigation_active.assert_not_called()


@pytest.mark.parametrize(
    ("clear_success", "expected_status", "expected_errors"),
    [
        (True, CurrentAction.FINISHED, ["ACTION_FAILED"]),
        (False, CurrentAction.FINISHED, ["NAVIGATION_ERROR", "ACTION_FAILED"]),
    ],
)
def test_cancel_order_clears_errors_after_navigation_cancel(
    clear_success, expected_status, expected_errors
):
    cancel_action_state = CurrentAction(
        action_id="cancel-clear", action_status=CurrentAction.RUNNING)
    navigation_error = Error(error_type="NAVIGATION_ERROR")
    action_error = Error(error_type="ACTION_FAILED")
    service_future = SimpleNamespace(add_done_callback=MagicMock())
    clear_error_client = SimpleNamespace(
        service_is_ready=MagicMock(return_value=True),
        call_async=MagicMock(return_value=service_future),
    )
    controller = SimpleNamespace(
        _cancel_action=Action(
            action_id="cancel-clear", action_type="cancelOrder"),
        _cancel_clear_error_pending=False,
        _cancel_action_failed=False,
        _cancel_action_failure_description="",
        _current_state=OrderState(
            action_states=[cancel_action_state],
            errors=[navigation_error, action_error],
        ),
        _process_vda_action_goal_handle_dict={},
        _is_navigation_active=MagicMock(return_value=False),
        _clear_error_svc_cli=clear_error_client,
        _visualization_state_request_generation=3,
        _visualization_state_request_pending=True,
        _visualization_state_request_started_at=10.0,
        _current_order=Order(order_id="order-clear"),
        _current_node_goal=object(),
        _current_node_actions=[object()],
        _publish_state=MagicMock(),
        logger=MagicMock(),
    )

    def update_state(partial_state):
        for field, value in partial_state.items():
            setattr(controller._current_state, field, value)

    def update_action_status(action_id, status, result_description=""):
        cancel_action_state.action_status = status
        cancel_action_state.result_description = result_description

    controller._get_action_status = MagicMock(return_value=CurrentAction.RUNNING)
    controller._update_state = MagicMock(side_effect=update_state)
    controller._update_action_status = MagicMock(side_effect=update_action_status)
    controller._start_cancel_order_error_clear = (
        lambda: VDA5050Controller._start_cancel_order_error_clear(controller)
    )
    controller._cancel_order_error_clear_response_callback = (
        lambda action_id, future:
        VDA5050Controller._cancel_order_error_clear_response_callback(
            controller, action_id, future
        )
    )
    controller._finish_cancel_order = (
        lambda action_id, clear_succeeded, result_description="":
        VDA5050Controller._finish_cancel_order(
            controller, action_id, clear_succeeded, result_description
        )
    )

    VDA5050Controller._cancel_order(controller)

    assert controller._cancel_clear_error_pending is True
    assert controller._cancel_action is not None
    service_future.add_done_callback.assert_called_once()
    clear_callback = service_future.add_done_callback.call_args.args[0]
    response_future = SimpleNamespace(
        result=lambda: SimpleNamespace(
            success=clear_success,
            message="cleared" if clear_success else "still in error",
        )
    )
    clear_callback(response_future)

    assert cancel_action_state.action_status == expected_status
    assert cancel_action_state.result_description == ""
    assert [error.error_type for error in controller._current_state.errors] == expected_errors
    assert controller._cancel_action is None
    assert controller._cancel_clear_error_pending is False
    assert controller._current_order.order_id == "-1"
    expected_generation = 4 if clear_success else 3
    assert controller._visualization_state_request_generation == expected_generation
    assert controller._visualization_state_request_pending is (not clear_success)
    controller._publish_state.assert_called_once_with()


@pytest.mark.parametrize("clear_success", [True, False])
def test_cancel_order_without_active_order_still_clears_errors(clear_success):
    cancel_action_state = CurrentAction(
        action_id="cancel-no-order", action_status=CurrentAction.WAITING)
    navigation_error = Error(error_type="NAVIGATION_ERROR")
    service_future = SimpleNamespace(add_done_callback=MagicMock())
    clear_error_client = SimpleNamespace(
        service_is_ready=MagicMock(return_value=True),
        call_async=MagicMock(return_value=service_future),
    )
    controller = SimpleNamespace(
        _cancel_action=Action(
            action_id="cancel-no-order", action_type="cancelOrder"),
        _cancel_attempted_generation=0,
        _cancel_clear_error_pending=False,
        _cancel_action_failed=False,
        _cancel_action_failure_description="",
        _order_generation=0,
        _instant_action_ids={"cancel-no-order"},
        _current_state=OrderState(
            action_states=[cancel_action_state],
            errors=[navigation_error],
        ),
        _process_vda_action_goal_handle_dict={},
        _is_navigation_active=MagicMock(return_value=False),
        _clear_error_svc_cli=clear_error_client,
        _visualization_state_request_generation=2,
        _visualization_state_request_pending=False,
        _visualization_state_request_started_at=None,
        _current_order=Order(order_id="-1"),
        _current_node_goal=None,
        _current_node_actions=[],
        _publish_state=MagicMock(),
        logger=MagicMock(),
    )

    def update_state(partial_state, publish_now=False):
        for field, value in partial_state.items():
            setattr(controller._current_state, field, value)

    def update_action_status(action_id, status, result_description=""):
        cancel_action_state.action_status = status
        if result_description:
            cancel_action_state.result_description = result_description

    controller._get_action_status = lambda action_id: cancel_action_state.action_status
    controller._has_current_order = MagicMock(return_value=False)
    controller._update_state = update_state
    controller._update_action_status = update_action_status
    controller._start_cancel_order_error_clear = (
        lambda: VDA5050Controller._start_cancel_order_error_clear(controller)
    )
    controller._cancel_order_error_clear_response_callback = (
        lambda action_id, future:
        VDA5050Controller._cancel_order_error_clear_response_callback(
            controller, action_id, future
        )
    )
    controller._finish_cancel_order = (
        lambda action_id, clear_succeeded, result_description="":
        VDA5050Controller._finish_cancel_order(
            controller, action_id, clear_succeeded, result_description
        )
    )

    VDA5050Controller._cancel_order(controller)

    assert cancel_action_state.action_status == CurrentAction.FAILED
    assert cancel_action_state.result_description == "There is no active order running."
    assert controller._cancel_clear_error_pending is True
    clear_callback = service_future.add_done_callback.call_args.args[0]
    clear_callback(SimpleNamespace(result=lambda: SimpleNamespace(
        success=clear_success,
        message="No error was active." if clear_success else "Still in error.",
    )))

    assert cancel_action_state.action_status == CurrentAction.FAILED
    assert cancel_action_state.result_description == "There is no active order running."
    assert controller._current_state.errors == ([] if clear_success else [navigation_error])
    assert controller._cancel_action is None


def test_visualization_request_timeout_retries_and_ignores_late_response():
    old_future = SimpleNamespace(add_done_callback=MagicMock())
    new_future = SimpleNamespace(add_done_callback=MagicMock())
    state_client = MagicMock()
    state_client.call_async.side_effect = [old_future, new_future]
    visualization_publisher = MagicMock()
    controller = SimpleNamespace(
        _visualization_state_request_pending=False,
        _visualization_state_request_started_at=None,
        _visualization_state_request_generation=0,
        _visualization_state_timeout=3.0,
        _current_visualization=SimpleNamespace(header_id=0, timestamp=""),
        _get_adapter_state_svc_cli=state_client,
        _publish_visualization_to_mc=visualization_publisher,
        _update_state_from_adapter=MagicMock(),
        logger=MagicMock(),
    )
    controller._publish_visualization_after_state = (
        lambda future, request_generation=None:
        VDA5050Controller._publish_visualization_after_state(
            controller, future, request_generation=request_generation
        )
    )

    VDA5050Controller._publish_visualization(controller)
    old_callback = old_future.add_done_callback.call_args.args[0]
    assert controller._visualization_state_request_pending

    controller._visualization_state_request_started_at = time.monotonic() - 4.0
    VDA5050Controller._publish_visualization(controller)
    new_callback = new_future.add_done_callback.call_args.args[0]

    assert state_client.call_async.call_count == 2
    assert controller._visualization_state_request_pending
    assert controller._visualization_state_request_generation == 2

    late_response = SimpleNamespace(result=MagicMock(return_value="stale-state"))
    old_callback(late_response)
    controller._update_state_from_adapter.assert_not_called()
    assert controller._visualization_state_request_pending

    current_response = SimpleNamespace(result=MagicMock(return_value="current-state"))
    new_callback(current_response)
    controller._update_state_from_adapter.assert_called_once_with("current-state")
    visualization_publisher.publish.assert_called_once_with(
        controller._current_visualization)
    assert controller._visualization_state_request_pending is False


@pytest.mark.parametrize(
    ("callback_name", "action_type"),
    [
        ("_navigate_to_node_result_callback", NavigateToNode),
        ("_navigate_through_nodes_result_callback", NavigateThroughNodes),
    ],
)
def test_canceled_navigation_is_not_reported_as_node_arrival(callback_name, action_type):
    controller = SimpleNamespace(
        _order_generation=2,
        _nav_through_nodes_goal_lock=threading.Lock(),
        _nav_through_nodes_goal_order_generation=2,
        _canceling_order=lambda: False,
        _navigate_to_node_goal_handle=object(),
        _navigate_to_node_goal_pending=False,
        _navigate_through_nodes_goal_handle=object(),
        _handle_navigation_failure=MagicMock(),
        _get_drivable_segment=MagicMock(),
        _nav_through_nodes_extensions_pending=0,
    )
    callback = getattr(VDA5050Controller, callback_name)

    callback(
        controller,
        action_result_future(action_type, GoalStatus.STATUS_CANCELED),
        order_generation=2,
    )

    controller._handle_navigation_failure.assert_called_once_with()
    controller._get_drivable_segment.assert_not_called()


@pytest.mark.parametrize(
    ("mutation", "expected_description"),
    [
        ("sequence_id", "sequenceId"),
        ("edge_endpoint", "endpoints"),
        ("edge_count", "fewer edge"),
        ("release_gap", "contiguous order prefix"),
        ("duplicate_action", "Duplicate order action"),
        ("edge_action", "does not support edge-scoped action execution"),
        ("unreleased_first_node", "first order node must be released"),
    ],
)
def test_order_msg_is_valid_rejects_invalid_graph(mutation, expected_description):
    order = get_order_new()
    if mutation == "sequence_id":
        order.edges[0].sequence_id = 0
    elif mutation == "edge_endpoint":
        order.edges[0].end_node_id = "unknown-node"
    elif mutation == "edge_count":
        order.edges.pop()
    elif mutation == "release_gap":
        order.edges[0].released = False
        order.edges[1].released = True
    elif mutation == "duplicate_action":
        order.nodes[0].actions = [
            Action(action_id="duplicate", action_type="pick",
                   blocking_type=Action.NONE)
        ]
        order.nodes[1].actions = [
            Action(action_id="duplicate", action_type="drop",
                   blocking_type=Action.NONE)
        ]
    elif mutation == "edge_action":
        order.edges[0].actions = [
            Action(action_id="edge-action", action_type="setLight",
                   blocking_type=Action.NONE)
        ]
    else:
        order.nodes[0].released = False

    is_valid, error = VDA5050Controller.order_msg_is_valid(None, order)

    assert is_valid is False
    assert error.error_type == "validationOrder"
    assert expected_description in error.error_description
    assert error.error_level == error.WARNING
    assert error.error_references[0].reference_value == order.order_id


def test_instant_action_msg_is_valid_accepts_none_blocking_action():
    instant_actions = InstantActions(
        actions=[
            Action(
                action_id="instant-1",
                action_type="startPause",
                blocking_type=Action.NONE,
            )
        ]
    )

    is_valid, error = VDA5050Controller.instant_action_msg_is_valid(
        None, instant_actions
    )

    assert is_valid is True
    assert error.error_description == ""


def test_instant_action_msg_is_valid_rejects_non_none_blocking_type():
    instant_actions = InstantActions(
        actions=[
            Action(
                action_id="instant-1",
                action_type="startPause",
                blocking_type=Action.HARD,
            )
        ]
    )

    is_valid, error = VDA5050Controller.instant_action_msg_is_valid(
        None, instant_actions
    )

    assert is_valid is False
    assert error.error_type == "invalidInstantAction"
    assert error.error_level == error.WARNING
    assert error.error_references[0].reference_value == "instant-1"


def test_instant_action_msg_is_valid_rejects_duplicate_action_ids():
    instant_actions = InstantActions(
        actions=[
            Action(action_id="duplicate", action_type="startPause",
                   blocking_type=Action.NONE),
            Action(action_id="duplicate", action_type="stopPause",
                   blocking_type=Action.NONE),
        ]
    )

    is_valid, error = VDA5050Controller.instant_action_msg_is_valid(
        None, instant_actions
    )

    assert is_valid is False
    assert "Duplicate" in error.error_description


def test_failed_navigation_extension_publishes_warning():
    class ControllerStub:
        def __init__(self):
            self.logger = MagicMock()
            self._current_state = OrderState(
                order_id="order-1", order_update_id=3)
            self.published_states = []
            self._nav_through_nodes_extensions_pending = 1
            self._pending_nav_through_nodes_result = None

        def _update_state(self, partial_state, publish_now=False):
            for field, value in partial_state.items():
                setattr(self._current_state, field, value)
            if publish_now:
                self.published_states.append(self._current_state)

        def _resume_pending_navigation_result(self):
            pass

    controller = ControllerStub()
    response = ExtendNavigation.Response(
        success=False,
        message="In-flight navigation extension is not supported",
    )
    future = SimpleNamespace(result=lambda: response)

    VDA5050Controller._extend_navigation_response_callback(
        controller, 8, future)

    assert controller._current_state.errors == []
    assert len(controller._current_state.informations) == 1
    info = controller._current_state.informations[0]
    assert info.info_type == "NAVIGATION_EXTENSION_DEFERRED"
    assert info.info_level == info.INFO
    assert "In-flight navigation extension is not supported" in info.info_description
    assert "current path will finish" in info.info_description
    assert controller.published_states == [controller._current_state]


def test_state_request_callback_accepts_action_without_result_description():
    class ControllerStub:
        def __init__(self):
            self._update_action_status = MagicMock()
            self._update_state_from_adapter = MagicMock()
            self._publish_state = MagicMock()

    controller = ControllerStub()
    action = Action(
        action_id="state-request",
        action_type="stateRequest",
        blocking_type=Action.NONE,
    )
    adapter_state = object()
    future = SimpleNamespace(result=lambda: adapter_state)

    # VDA Action 没有 result_description，回调仍应完成状态合并与发布。
    VDA5050Controller._get_state_from_adapter_callback(
        controller, action, future)

    controller._update_action_status.assert_called_once_with(
        action.action_id, CurrentAction.FINISHED)
    controller._update_state_from_adapter.assert_called_once_with(
        adapter_state)
    controller._publish_state.assert_called_once_with()


def get_order_update(order_id=str(uuid4()), order_update_id=0):
    return Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=order_update_id,
        nodes=[
            Node(
                node_id="node1",
                sequence_id=8,
                released=True,
                node_position=NodePosition(
                    x=2.0,
                    y=0.95,
                    theta=-0.66,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node2",
                sequence_id=10,
                released=True,
                node_position=NodePosition(
                    x=1.18,
                    y=-1.76,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
        ],
        edges=[
            Edge(
                edge_id="edge1",
                sequence_id=9,
                released=True,
                start_node_id="node1",
                end_node_id="node2",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            )
        ],
    )


def get_order_w_unreleased_new(order_id=str(uuid4()), order_update_id=0):
    return Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=order_update_id,
        nodes=[
            Node(
                node_id="node1",
                sequence_id=0,
                released=True,
                node_position=NodePosition(
                    x=2.0,
                    y=0.95,
                    theta=-0.66,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node2",
                sequence_id=2,
                released=True,
                node_position=NodePosition(
                    x=1.18,
                    y=-1.76,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node3",
                sequence_id=4,
                released=True,
                node_position=NodePosition(
                    x=-0.38,
                    y=1.89,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node4",
                sequence_id=6,
                released=True,
                node_position=NodePosition(
                    x=-0.17,
                    y=1.74,
                    theta=-2.6,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
            Node(
                node_id="node1",
                sequence_id=8,
                released=False,
                node_position=NodePosition(
                    x=2.0,
                    y=0.95,
                    theta=-0.66,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
            ),
        ],
        edges=[
            Edge(
                edge_id="edge1",
                sequence_id=1,
                released=True,
                start_node_id="node1",
                end_node_id="node2",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
            Edge(
                edge_id="edge2",
                sequence_id=3,
                released=True,
                start_node_id="node2",
                end_node_id="node3",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
            Edge(
                edge_id="edge3",
                sequence_id=5,
                released=True,
                start_node_id="node3",
                end_node_id="node4",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
            Edge(
                edge_id="edge4",
                sequence_id=7,
                released=False,
                start_node_id="node4",
                end_node_id="node1",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
        ],
    )


def get_stitch_orders(order_id=str(uuid4())):
    action1 = Action(
        action_type="foo",
        action_id=str(uuid4()),
        action_description="Foo description",
        blocking_type="NONE",
        action_parameters=[
            ActionParameter(key="foo", value="bar")
        ]
    )
    action2 = Action(
        action_type="bar",
        action_id=str(uuid4()),
        action_description="Bar description",
        blocking_type="NONE",
        action_parameters=[
            ActionParameter(key="foo", value="bar")
        ]
    )
    action3 = Action(
        action_type="foobar",
        action_id=str(uuid4()),
        action_description="FooBar description",
        blocking_type="NONE",
        action_parameters=[
            ActionParameter(key="abc", value="xyz")
        ]
    )
    base_order = Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=0,
        nodes=[
            Node(
                node_id="node1",
                sequence_id=0,
                released=True,
                node_position=NodePosition(
                    x=2.0,
                    y=0.95,
                    theta=-0.66,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
                actions=[
                    action1
                ],
            ),
            Node(
                node_id="node2",
                sequence_id=2,
                released=True,
                node_position=NodePosition(
                    x=1.18,
                    y=-1.76,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
                actions=[
                    action2
                ],
            ),
        ],
        edges=[
            Edge(
                edge_id="edge1",
                sequence_id=1,
                released=True,
                start_node_id="node1",
                end_node_id="node2",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
        ],
    )

    stitch_order = Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=1,
        nodes=[
            Node(
                node_id="node2",
                sequence_id=2,
                released=True,
                node_position=NodePosition(
                    x=1.18,
                    y=-1.76,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
                actions=[
                    action2
                ],
            ),
            Node(
                node_id="node3",
                sequence_id=4,
                released=True,
                node_position=NodePosition(
                    x=-0.38,
                    y=1.89,
                    theta=0.0,
                    allowed_deviation_x_y=0.0,
                    allowed_deviation_theta=0.0,
                    map_id="map",
                ),
                actions=[
                    action3
                ],
            ),
        ],
        edges=[
            Edge(
                edge_id="edge2",
                sequence_id=3,
                released=True,
                start_node_id="node2",
                end_node_id="node3",
                max_speed=10.0,
                max_height=10.0,
                min_height=1.0,
            ),
        ],
    )

    return [base_order, stitch_order]


def test_vda5050_controller_node_new_order(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):

    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    # add a spy to validate used navigation goal parameters
    spy_send_adapter_navigate_to_node = mocker.spy(
        node, "send_adapter_navigate_to_node"
    )

    # add a spy to validate accept order is called correctly
    spy_accept_order = mocker.spy(node, "_accept_order")

    # generate an order and let the node process it
    order_id = str(uuid4())
    order = get_order_new(order_id)
    node.process_order(order)
    assert node._order_generation == 1

    rclpy.spin_once(node)

    spy_accept_order.assert_called_once_with(
        order=order, mode=OrderAcceptModes.NEW)

    rclpy.spin_once(adapter_node)

    # check node states were properly updated
    assert node._current_order == order
    assert node._current_state.order_id == order_id
    assert node._current_state.order_update_id == 0

    # The order has 5 nodes and 4 edges but the first edge and node
    # are processed as soon as the order is accepted.
    assert len(node._current_state.node_states) == 4
    assert len(node._current_state.edge_states) == 4

    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 0

    # Assert the first navigation goal was sent to the adapter,
    # and that the parameters matches order's first edge and second node.
    # Note: the standard assumes the vehicle is on the first node already,
    # so the first navigation command is to the second order node.
    spy_send_adapter_navigate_to_node.assert_called_once_with(
        edge=order.edges[0], node=order.nodes[1]
    )

    # Future for invoking adapter navigation goal result callback
    future = succeeded_action_result_future(NavigateToNode)

    spy_send_adapter_navigate_to_node.reset_mock()
    # Simulate the adapter reached navigation goal
    node._navigate_to_node_result_callback(future)
    node._on_active_order()

    spy_send_adapter_navigate_to_node.assert_called_once_with(
        edge=order.edges[1], node=order.nodes[2]
    )

    assert len(node._current_state.node_states) == 3
    assert len(node._current_state.edge_states) == 3
    assert node._current_state.last_node_id == "node2"
    assert node._current_state.last_node_sequence_id == 2

    spy_send_adapter_navigate_to_node.reset_mock()
    # Simulate the adapter reached navigation goal
    node._navigate_to_node_result_callback(future)
    node._on_active_order()

    spy_send_adapter_navigate_to_node.assert_called_once_with(
        edge=order.edges[2], node=order.nodes[3]
    )

    assert len(node._current_state.node_states) == 2
    assert len(node._current_state.edge_states) == 2
    assert node._current_state.last_node_id == "node3"
    assert node._current_state.last_node_sequence_id == 4

    spy_send_adapter_navigate_to_node.reset_mock()
    # Simulate the adapter reached navigation goal
    node._navigate_to_node_result_callback(future)
    node._on_active_order()

    spy_send_adapter_navigate_to_node.assert_called_once_with(
        edge=order.edges[3], node=order.nodes[4]
    )
    assert len(node._current_state.node_states) == 1
    assert len(node._current_state.edge_states) == 1
    assert node._current_state.last_node_id == "node4"
    assert node._current_state.last_node_sequence_id == 6

    spy_send_adapter_navigate_to_node.reset_mock()
    # Simulate the adapter reached navigation goal
    node._navigate_to_node_result_callback(future)
    node._on_active_order()

    assert len(node._current_state.node_states) == 0
    assert len(node._current_state.edge_states) == 0
    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 8


def test_vda5050_controller_node_update_order(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    # add a spy to validate used navigation goal parameters
    spy_send_adapter_navigate_to_node = mocker.spy(
        node, "send_adapter_navigate_to_node"
    )

    # add a spy to validate accept order is called correctly
    spy_accept_order = mocker.spy(node, "_accept_order")

    # Send first new order
    order_id = str(uuid4())
    order = get_order_new(order_id)
    node.process_order(order)

    rclpy.spin_once(node)
    rclpy.spin_once(adapter_node)

    # Simulate the adapter reached navigation goals
    future = succeeded_action_result_future(NavigateToNode)

    # The NEW order contains 5 nodes and 4 edges. The first node (in deviation range)
    # is processed and remove, and 4 nodes are send to navigate to.
    node._navigate_to_node_result_callback(future)
    node._on_active_order()
    node._navigate_to_node_result_callback(future)
    node._on_active_order()
    node._navigate_to_node_result_callback(future)
    node._on_active_order()
    node._navigate_to_node_result_callback(future)
    node._on_active_order()
    # Finish initial order

    spy_accept_order.reset_mock()
    spy_send_adapter_navigate_to_node.reset_mock()

    # Send update order
    order = get_order_update(order_id, 1)  # Same order id
    node.process_order(order)
    node._on_active_order()

    spy_accept_order.assert_called_once_with(
        order=order, mode=OrderAcceptModes.UPDATE)
    assert node._order_generation == 2

    # check node states were properly updated
    assert node._current_order == order
    assert node._current_state.order_id == order_id
    assert node._current_state.order_update_id == 1

    # The order has 2 nodes and 1 edges but the first edge and node
    # are processed as soon as the order is accepted.
    assert len(node._current_state.node_states) == 1
    assert len(node._current_state.edge_states) == 1

    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 8

    # Assert the first navigation goal was sent to the adapter,
    # and that the parameters matches order's first edge and second node.
    # Note: the standard assumes the vehicle is on the first node already,
    # so the first navigation command is to the second order node.
    spy_send_adapter_navigate_to_node.assert_called_once_with(
        edge=order.edges[0], node=order.nodes[1]
    )

    # Future for invoking adapter navigation goal result callback
    future = succeeded_action_result_future(NavigateToNode)

    # Simulate the adapter reached navigation goal
    node._navigate_to_node_result_callback(future)
    assert len(node._current_state.node_states) == 0
    assert len(node._current_state.edge_states) == 0
    assert node._current_state.last_node_id == "node2"
    assert node._current_state.last_node_sequence_id == 10


def test_vda5050_controller_node_stitch_order(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    # add a spy to validate accept order is called correctly
    spy_accept_order = mocker.spy(node, "_accept_order")

    # Get the base order and the stitch order, both with two nodes
    # and one edge each. The resulting (stitched) order should contain
    # three nodes and two edges:
    # (A -> B) + (B -> C) = A -> B -> C
    # All three nodes have an action.
    [base_order, stitch_order] = get_stitch_orders()

    node.process_order(base_order)
    assert node._order_generation == 1

    rclpy.spin_once(node)
    rclpy.spin_once(adapter_node)

    # Simulate the adapter reached navigation goals
    future = Future()
    result_response = NavigateToNode.Impl.GetResultService.Response()
    result_response.status = 4  # GoalStatus.STATUS_SUCCEEDED
    future.set_result(result=result_response)

    # The base order contains 2 nodes and 1 edge. The first node (in deviation range)
    # is processed and removed, and 1 node is sent to navigate to.
    node._navigate_to_node_result_callback(future)

    assert len(node._current_order.nodes) == 2
    assert len(node._current_order.edges) == 1
    assert len(node._current_state.action_states) == 2

    spy_accept_order.reset_mock()
    # Process the stitching order
    node.process_order(stitch_order)
    spy_accept_order.assert_called_once_with(
        order=stitch_order, mode=OrderAcceptModes.STITCH)
    # STITCH 延伸同一订单图，不应使正在执行的导航回调失效。
    assert node._order_generation == 1

    assert node._current_state.order_id == stitch_order.order_id
    assert node._current_state.order_update_id == 1

    assert len(node._current_order.nodes) == 3
    assert len(node._current_order.edges) == 2

    # Stitch node (node2, seq=2) was already visited — dedup drops it
    assert len(node._current_state.node_states) == 1
    assert node._current_state.node_states[0].node_id == "node3"
    assert len(node._current_state.edge_states) == 1
    assert len(node._current_state.action_states) == 3

    assert node._current_state.last_node_id == "node2"
    assert node._current_state.last_node_sequence_id == 2

    node._navigate_to_node_result_callback(future)
    assert len(node._current_state.node_states) == 0
    assert len(node._current_state.edge_states) == 0
    assert len(node._current_state.action_states) == 3

    assert node._current_state.last_node_id == "node3"
    assert node._current_state.last_node_sequence_id == 4


def test_vda5050_controller_zero_based_stitch_order(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    """Zero-based updates are rebased before stitch validation."""
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    [base_order, stitch_order] = get_stitch_orders()
    stitch_order.nodes[0].sequence_id = 0
    stitch_order.nodes[1].sequence_id = 2
    stitch_order.edges[0].sequence_id = 1

    node.process_order(base_order)

    future = succeeded_action_result_future(NavigateToNode)
    node._navigate_to_node_result_callback(future)

    node.process_order(stitch_order)

    assert stitch_order.nodes[0].sequence_id == 2
    assert stitch_order.nodes[1].sequence_id == 4
    assert stitch_order.edges[0].sequence_id == 3
    assert node._current_order.nodes[-1].node_id == "node3"
    assert node._current_order.nodes[-1].sequence_id == 4


def test_vda5050_controller_node_reject_order(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    # add a spy to validate that the order has been rejected
    spy_reject_order = mocker.spy(node, "_reject_order")

    # UPDATE test fail - lower order_update_id

    # Send first new order
    order_id = str(uuid4())
    order = get_order_new(order_id, 1)
    node.process_order(order)

    # Simulate the adapter reached navigation goals
    future = succeeded_action_result_future(NavigateToNode)

    # The NEW order contains 5 nodes and 4 edges. The first node (in deviation range)
    # is processed and remove, and 4 nodes are send to navigate to.
    node._navigate_to_node_result_callback(future)
    node._navigate_to_node_result_callback(future)
    node._navigate_to_node_result_callback(future)
    node._navigate_to_node_result_callback(future)
    # Finish initial order

    spy_reject_order.reset_mock()

    # Same order id, lower order_update_id
    order = get_order_update(order_id, 0)
    node.process_order(order)

    spy_reject_order.assert_called_once_with(
        order=order,
        error=OrderRejectErrors.ORDER_UPDATE_ERROR,
        description="New update id 0 lower than old update id 1",
    )


def get_instant_actions(*action_types):
    """Build an InstantActions message with one action per given action_type."""
    return InstantActions(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        actions=[
            Action(
                action_type=action_type,
                action_id=str(uuid4()),
                action_description=action_type,
                blocking_type="NONE",
            )
            for action_type in action_types
        ],
    )


def test_instant_action_does_not_block_active_order_check(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    # No order and no instant actions: no active order
    assert not node._has_current_order()

    # The instant action is now WAITING/INITIALIZING — _has_current_order must still be False
    instant_actions = get_instant_actions("startPause")
    node.process_instant_actions(instant_actions)
    assert not node._has_current_order()

    # Manually move the instant action to RUNNING to cover that state too
    action_id = instant_actions.actions[0].action_id
    node._update_action_status(action_id, CurrentAction.RUNNING)
    assert not node._has_current_order()

    # Once it finishes it should still return False (no nodes/edges)
    node._update_action_status(action_id, CurrentAction.FINISHED)
    assert not node._has_current_order()


def test_instant_action_ids_tracked_on_process_instant_actions(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    node = VDA5050Controller()

    instant_actions = get_instant_actions("startPause", "stopPause")
    node.process_instant_actions(instant_actions)

    for action in instant_actions.actions:
        assert action.action_id in node._instant_action_ids


def test_order_action_still_blocks_active_order_check(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    # Process an order that contains a node action
    [base_order, _] = get_stitch_orders()
    node.process_order(base_order)

    rclpy.spin_once(node)
    rclpy.spin_once(adapter_node)

    # The order has running node/edge states — active order must be True
    assert node._has_current_order()

    # Layering an instant action on top must not change the result
    instant_actions = get_instant_actions("startPause")
    node.process_instant_actions(instant_actions)
    assert node._has_current_order()


def test_vda5050_controller_node_new_order_nav_through_nodes(
    mocker,
    adapter_node,
    action_server_nav_through_nodes,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    """Tests the original node list with the nav through nodes functionality."""
    nav_through_nodes_param = Parameter(
        "enable_nav_through_nodes", type_=Parameter.Type.BOOL, value=True)
    node = VDA5050Controller(parameter_overrides=[nav_through_nodes_param])
    node.logger.set_level(LoggingSeverity.DEBUG)

    # add a spy to validate used navigation goal parameters
    spy_send_adapter_navigate_through_nodes = mocker.spy(
        node, "send_adapter_navigate_through_nodes"
    )

    spy_process_last_edge_node = mocker.spy(
        node, "_process_last_edge_node"
    )

    # add a spy to validate accept order is called correctly
    spy_accept_order = mocker.spy(node, "_accept_order")

    # generate an order and let the node process it
    order_id = str(uuid4())
    order = get_order_new(order_id)
    node.process_order(order)

    rclpy.spin_once(node)

    spy_accept_order.assert_called_once_with(
        order=order, mode=OrderAcceptModes.NEW)

    rclpy.spin_once(adapter_node)

    # check node states were properly updated
    assert node._current_order == order
    assert node._current_state.order_id == order_id
    assert node._current_state.order_update_id == 0

    # The order has 5 nodes and 4 edges but the first edge and node
    # are processed as soon as the order is accepted.
    assert len(node._current_state.node_states) == 4
    assert len(node._current_state.edge_states) == 4

    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 0

    # Assert the navigation through nodes goal was sent to the adapter,
    # and that the parameters matches order's released edges
    # Note: the standard assumes the vehicle is on the first node already,
    # so the first navigation command is to the second order node.

    spy_send_adapter_navigate_through_nodes.assert_called_once_with(
        edges=order.edges[:4], nodes=order.nodes[1:5]
    )

    feedback_msg = NavigateThroughNodes.Impl.FeedbackMessage()
    # Check that a feedback of the current node doesn't affect the current state
    feedback_msg.feedback.last_node = order.nodes[0]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_not_called()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 4
    assert len(node._current_state.edge_states) == 4
    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 0

    # Next node has been reached and a feedback message is published
    feedback_msg.feedback.last_node = order.nodes[1]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 3
    assert len(node._current_state.edge_states) == 3
    assert node._current_state.last_node_id == "node2"
    assert node._current_state.last_node_sequence_id == 2

    # Next node has been reached and a feedback message is published
    feedback_msg.feedback.last_node = order.nodes[2]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 2
    assert len(node._current_state.edge_states) == 2
    assert node._current_state.last_node_id == "node3"
    assert node._current_state.last_node_sequence_id == 4

    # Next node has been reached and a feedback message is published
    feedback_msg.feedback.last_node = order.nodes[3]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 1
    assert len(node._current_state.edge_states) == 1
    assert node._current_state.last_node_id == "node4"
    assert node._current_state.last_node_sequence_id == 6

    # Last node reached as indicated by the result coming through
    # A feedback message shouldn't be published for the final node in a navigation order

    # Simulate the adapter reached navigation goals
    future = succeeded_action_result_future(NavigateThroughNodes)
    node._navigate_through_nodes_result_callback(future)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()
    assert len(node._current_state.node_states) == 0
    assert len(node._current_state.edge_states) == 0
    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 8


def test_vda5050_controller_node_new_order_nav_through_nodes_unreleased_nodes(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_nav_through_nodes,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    """Tests with some unreleased nodes on the horizon and navigate through nodes."""
    nav_through_nodes_param = Parameter(
        "enable_nav_through_nodes", type_=Parameter.Type.BOOL, value=True)
    node = VDA5050Controller(parameter_overrides=[nav_through_nodes_param])
    node.logger.set_level(LoggingSeverity.DEBUG)

    # add a spy to validate used navigation goal parameters
    spy_send_adapter_navigate_through_nodes = mocker.spy(
        node, "send_adapter_navigate_through_nodes"
    )

    spy_process_last_edge_node = mocker.spy(
        node, "_process_last_edge_node"
    )

    # add a spy to validate accept order is called correctly
    spy_accept_order = mocker.spy(node, "_accept_order")

    # generate an order and let the node process it
    order_id = str(uuid4())
    order = get_order_w_unreleased_new(order_id)
    node.process_order(order)

    rclpy.spin_once(node)

    spy_accept_order.assert_called_once_with(
        order=order, mode=OrderAcceptModes.NEW)

    rclpy.spin_once(adapter_node)

    # check node states were properly updated
    assert node._current_order == order
    assert node._current_state.order_id == order_id
    assert node._current_state.order_update_id == 0

    # The order has 5 nodes and 4 edges but the first edge and node
    # are processed as soon as the order is accepted.
    assert len(node._current_state.node_states) == 4
    assert len(node._current_state.edge_states) == 4

    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 0

    # Assert the first navigation goal was sent to the adapter,
    # and that the parameters matches order's first edge and second node.
    # Note: the standard assumes the vehicle is on the first node already,
    # so the first navigation command is to the second order node.

    spy_send_adapter_navigate_through_nodes.assert_called_once_with(
        edges=order.edges[:3], nodes=order.nodes[1:4]
    )

    feedback_msg = NavigateThroughNodes.Impl.FeedbackMessage()
    # Check that a feedback of the current node doesn't affect the current state
    feedback_msg.feedback.last_node = order.nodes[0]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_not_called()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 4
    assert len(node._current_state.edge_states) == 4
    assert node._current_state.last_node_id == "node1"
    assert node._current_state.last_node_sequence_id == 0

    # Next node has been reached and a feedback message is published
    feedback_msg.feedback.last_node = order.nodes[1]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 3
    assert len(node._current_state.edge_states) == 3
    assert node._current_state.last_node_id == "node2"
    assert node._current_state.last_node_sequence_id == 2

    # Next node has been reached and a feedback message is published
    feedback_msg.feedback.last_node = order.nodes[2]
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()

    assert len(node._current_state.node_states) == 2
    assert len(node._current_state.edge_states) == 2
    assert node._current_state.last_node_id == "node3"
    assert node._current_state.last_node_sequence_id == 4

    # Last node reached as indicated by the result coming through
    # A feedback message shouldn't be published for the final node in a navigation order

    # Simulate the adapter reached navigation goals
    future = succeeded_action_result_future(NavigateThroughNodes)
    node._navigate_through_nodes_result_callback(future)

    spy_process_last_edge_node.assert_called_once()
    spy_process_last_edge_node.reset_mock()
    assert len(node._current_state.node_states) == 1
    assert len(node._current_state.edge_states) == 1
    assert node._current_state.last_node_id == "node4"
    assert node._current_state.last_node_sequence_id == 6

    # This is the final node with a released horizon.
    # Therefore a request should be flagged on the next tick
    node._on_active_order()
    assert node._current_state.new_base_request is True


@pytest.mark.parametrize(
    ("extension_success", "goal_generation_matches"),
    [(True, True), (False, True), (True, False)],
)
def test_vda5050_controller_stitch_while_navigating(
    mocker,
    adapter_node,
    action_server_nav_through_nodes,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
    extension_success,
    goal_generation_matches,
):
    """
    Extend the in-flight navigation goal when a stitch order arrives.

    This should use ExtendNavigation instead of sending a goal that would be rejected.
    """
    nav_through_nodes_param = Parameter(
        "enable_nav_through_nodes", type_=Parameter.Type.BOOL, value=True)
    node = VDA5050Controller(parameter_overrides=[nav_through_nodes_param])
    node.logger.set_level(LoggingSeverity.DEBUG)

    spy_send_adapter_navigate_through_nodes = mocker.spy(
        node, "send_adapter_navigate_through_nodes"
    )
    spy_accept_order = mocker.spy(node, "_accept_order")
    spy_extend_nav = mocker.spy(node, "_extend_navigation_response_callback")

    # Mock the extend_nav service client so we don't rely on DDS discovery
    from vda5050_connector.srv import ExtendNavigation
    mock_extend_response = ExtendNavigation.Response(
        success=extension_success, message="ok")
    mock_extend_future = Future()
    node._extend_nav_svc_cli = mocker.MagicMock()
    node._extend_nav_svc_cli.service_is_ready = mocker.MagicMock(
        return_value=True)
    node._extend_nav_svc_cli.call_async = mocker.MagicMock(
        return_value=mock_extend_future)

    # --- Build a base order with 3 released nodes + 1 unreleased horizon node ---
    order_id = str(uuid4())
    base_order = Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=0,
        nodes=[
            Node(node_id="A", sequence_id=0, released=True,
                 node_position=NodePosition(x=0.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="B", sequence_id=2, released=True,
                 node_position=NodePosition(x=1.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="C", sequence_id=4, released=True,
                 node_position=NodePosition(x=2.0, y=0.0, theta=0.0, map_id="map")),
            # Horizon (unreleased)
            Node(node_id="D", sequence_id=6, released=False,
                 node_position=NodePosition(x=3.0, y=0.0, theta=0.0, map_id="map")),
        ],
        edges=[
            Edge(edge_id="e1", sequence_id=1, released=True,
                 start_node_id="A", end_node_id="B"),
            Edge(edge_id="e2", sequence_id=3, released=True,
                 start_node_id="B", end_node_id="C"),
            # Horizon (unreleased)
            Edge(edge_id="e3", sequence_id=5, released=False,
                 start_node_id="C", end_node_id="D"),
        ],
    )

    # Process the base order
    node.process_order(base_order)
    rclpy.spin_once(node)
    spy_accept_order.assert_called_once_with(
        order=base_order, mode=OrderAcceptModes.NEW)
    rclpy.spin_once(adapter_node)
    rclpy.spin_once(node)

    # The controller should navigate through released nodes B and C (2 edges)
    spy_send_adapter_navigate_through_nodes.assert_called_once_with(
        edges=base_order.edges[:2], nodes=base_order.nodes[1:3]
    )
    assert node._nav_through_nodes_last_seq == 4  # seq of node C
    # 本测试将 fake Adapter 的已接受 goal 作为在途续接前提。
    node._navigate_through_nodes_goal_handle = object()
    node._nav_through_nodes_goal_order_generation = (
        node._order_generation
        if goal_generation_matches
        else node._order_generation - 1
    )

    assert node._current_state.last_node_id == "A"
    assert node._current_state.last_node_sequence_id == 0
    assert len(node._current_state.node_states) == 3  # B, C, D (horizon)
    assert len(node._current_state.edge_states) == 3

    # --- Robot reaches node B (feedback) ---
    feedback_msg = NavigateThroughNodes.Impl.FeedbackMessage()
    feedback_msg.feedback.last_node = base_order.nodes[1]  # B
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    assert node._current_state.last_node_id == "B"
    assert node._current_state.last_node_sequence_id == 2
    assert len(node._current_state.node_states) == 2  # C, D

    # --- Stitch order arrives while robot is between B and C ---
    # Extends the base: releases D and adds E
    spy_accept_order.reset_mock()
    spy_send_adapter_navigate_through_nodes.reset_mock()

    stitch_order = Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=1,
        nodes=[
            # Stitch node (must match last released node of old base)
            Node(node_id="C", sequence_id=4, released=True,
                 node_position=NodePosition(x=2.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="D", sequence_id=6, released=True,
                 node_position=NodePosition(x=3.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="E", sequence_id=8, released=True,
                 node_position=NodePosition(x=4.0, y=0.0, theta=0.0, map_id="map")),
        ],
        edges=[
            Edge(edge_id="e3", sequence_id=5, released=True,
                 start_node_id="C", end_node_id="D"),
            Edge(edge_id="e4", sequence_id=7, released=True,
                 start_node_id="D", end_node_id="E"),
        ],
    )

    node.process_order(stitch_order)
    spy_accept_order.assert_called_once_with(
        order=stitch_order, mode=OrderAcceptModes.STITCH)

    if not goal_generation_matches:
        # 旧订单 goal 仍在途时不应把新订单更新续接到它上面。
        node._extend_nav_svc_cli.call_async.assert_not_called()
        spy_send_adapter_navigate_through_nodes.assert_not_called()
        assert node._current_order.order_update_id == 1
        return

    # The controller should NOT send a new navigation goal (would be rejected)
    spy_send_adapter_navigate_through_nodes.assert_not_called()

    # It should have called extend_nav_svc_cli.call_async
    node._extend_nav_svc_cli.call_async.assert_called_once()
    req = node._extend_nav_svc_cli.call_async.call_args[0][0]
    # nodes[0] is the stitch reference node (C), plus 2 new nodes (D, E)
    assert len(req.nodes) == 3
    assert req.nodes[0].node_id == "C"
    assert req.nodes[1].node_id == "D"
    assert req.nodes[2].node_id == "E"
    assert len(req.edges) == 2

    assert node._current_state.last_node_id == "B"
    assert node._current_state.last_node_sequence_id == 2

    # --- Robot reaches node C (feedback) ---
    feedback_msg.feedback.last_node = base_order.nodes[2]  # C
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    assert node._current_state.last_node_id == "C"
    assert node._current_state.last_node_sequence_id == 4

    # Simulate the original goal finishing before the STITCH service replies.
    future = Future()
    result_response = NavigateThroughNodes.Impl.GetResultService.Response()
    result_response.status = 4  # GoalStatus.STATUS_SUCCEEDED
    future.set_result(result=result_response)
    node._navigate_through_nodes_result_callback(future)

    assert node._pending_nav_through_nodes_result is future
    assert node._current_state.last_node_id == "C"
    # A fallback goal must wait until the adapter's extension callback releases its lock.
    spy_send_adapter_navigate_through_nodes.assert_not_called()

    mock_extend_future.set_result(mock_extend_response)
    spy_extend_nav.assert_called_once()

    if extension_success:
        # Successful extension advances the original goal through E.
        assert node._nav_through_nodes_last_seq == 8
        assert node._current_state.last_node_id == "E"
        assert node._current_state.last_node_sequence_id == 8
        assert len(node._current_state.node_states) == 0
        assert len(node._current_state.edge_states) == 0
        spy_send_adapter_navigate_through_nodes.assert_not_called()
    else:
        # Rejected extension dispatches the remaining D->E segment only after its response.
        spy_send_adapter_navigate_through_nodes.assert_called_once()
        fallback_call = spy_send_adapter_navigate_through_nodes.call_args.kwargs
        assert [node.node_id for node in fallback_call["nodes"]] == ["D", "E"]


def test_vda5050_controller_skipped_feedback_nodes(
    mocker,
    adapter_node,
    action_server_nav_through_nodes,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    """
    Process skipped nodes individually when intermediate feedback is missed.

    The result callback must handle nodes C, D, and E in sequence.
    """
    nav_through_nodes_param = Parameter(
        "enable_nav_through_nodes", type_=Parameter.Type.BOOL, value=True)
    node = VDA5050Controller(parameter_overrides=[nav_through_nodes_param])
    node.logger.set_level(LoggingSeverity.DEBUG)

    spy_process_last_edge_node = mocker.spy(node, "_process_last_edge_node")
    spy_process_node = mocker.spy(node, "_process_node")

    # --- Order: A(0) → B(2) → C(4) → D(6) → E(8) ---
    order_id = str(uuid4())
    order = Order(
        header_id=0,
        timestamp=get_vda5050_ts(),
        version="1.1.1",
        manufacturer="MANUFACTURER",
        serial_number="SERIAL_NUMBER",
        order_id=order_id,
        order_update_id=0,
        nodes=[
            Node(node_id="A", sequence_id=0, released=True,
                 node_position=NodePosition(x=0.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="B", sequence_id=2, released=True,
                 node_position=NodePosition(x=1.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="C", sequence_id=4, released=True,
                 node_position=NodePosition(x=2.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="D", sequence_id=6, released=True,
                 node_position=NodePosition(x=3.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="E", sequence_id=8, released=True,
                 node_position=NodePosition(x=4.0, y=0.0, theta=0.0, map_id="map")),
        ],
        edges=[
            Edge(edge_id="e1", sequence_id=1, released=True,
                 start_node_id="A", end_node_id="B"),
            Edge(edge_id="e2", sequence_id=3, released=True,
                 start_node_id="B", end_node_id="C"),
            Edge(edge_id="e3", sequence_id=5, released=True,
                 start_node_id="C", end_node_id="D"),
            Edge(edge_id="e4", sequence_id=7, released=True,
                 start_node_id="D", end_node_id="E"),
        ],
    )

    # Process the order — robot is at A, nav goal sent for B→C→D→E
    node.process_order(order)
    rclpy.spin_once(node)
    rclpy.spin_once(adapter_node)

    assert node._current_state.last_node_id == "A"
    assert node._current_state.last_node_sequence_id == 0
    assert node._nav_through_nodes_last_seq == 8
    assert len(node._current_state.node_states) == 4  # B, C, D, E
    assert len(node._current_state.edge_states) == 4

    # --- Feedback: robot reaches B (seq 2) ---
    feedback_msg = NavigateThroughNodes.Impl.FeedbackMessage()
    feedback_msg.feedback.last_node = order.nodes[1]  # B
    node._navigate_through_nodes_feedback_callback(feedback_msg)

    assert node._current_state.last_node_id == "B"
    assert node._current_state.last_node_sequence_id == 2
    assert len(node._current_state.node_states) == 3  # C, D, E
    assert len(node._current_state.edge_states) == 3

    # --- No feedback for C or D (missed / proximity threshold not met) ---

    # Reset spies before result callback to count only the skipped-node processing
    spy_process_last_edge_node.reset_mock()
    spy_process_node.reset_mock()

    # Spy on _publish_state to verify each node triggers a separate state message
    spy_publish_state = mocker.spy(node, "_publish_state")

    # --- Result: navigation goal completes (robot arrived at E) ---
    future = Future()
    result_response = NavigateThroughNodes.Impl.GetResultService.Response()
    result_response.status = 4  # GoalStatus.STATUS_SUCCEEDED
    future.set_result(result=result_response)
    node._navigate_through_nodes_result_callback(future)

    # The result callback should have consumed C, D, E individually
    assert spy_process_last_edge_node.call_count == 3
    assert spy_process_node.call_count == 3

    # Each node must publish its own state message — not batched into one
    assert spy_publish_state.call_count == 3

    # Verify _process_node was called in order: C, D, E
    process_node_calls = [call.kwargs.get("node") or call.args[0]
                          for call in spy_process_node.call_args_list]
    assert process_node_calls[0].node_id == "C"
    assert process_node_calls[0].sequence_id == 4
    assert process_node_calls[1].node_id == "D"
    assert process_node_calls[1].sequence_id == 6
    assert process_node_calls[2].node_id == "E"
    assert process_node_calls[2].sequence_id == 8

    # Final state: all nodes consumed, robot at E
    assert node._current_state.last_node_id == "E"
    assert node._current_state.last_node_sequence_id == 8
    assert len(node._current_state.node_states) == 0
    assert len(node._current_state.edge_states) == 0


def test_stitch_preserves_action_statuses(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    """
    Preserve action statuses across a stitch update.

    Finished and running actions are retained; only new actions are appended.
    """
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    [base_order, stitch_order] = get_stitch_orders()
    action1 = base_order.nodes[0].actions[0]
    action2 = base_order.nodes[1].actions[0]
    action3 = stitch_order.nodes[1].actions[0]

    # Accept the base order
    node.process_order(base_order)
    rclpy.spin_once(node)
    rclpy.spin_once(adapter_node)

    # Simulate robot reaching node2 via _process_last_edge_node
    node._process_last_edge_node()

    assert len(node._current_state.action_states) == 2

    # Simulate action progression: action1 finished, action2 running
    node._update_action_status(action1.action_id, CurrentAction.FINISHED)
    node._update_action_status(action2.action_id, CurrentAction.RUNNING)

    # Verify pre-stitch state
    action_map = {a.action_id: a for a in node._current_state.action_states}
    assert action_map[action1.action_id].action_status == CurrentAction.FINISHED
    assert action_map[action2.action_id].action_status == CurrentAction.RUNNING

    # Process the stitch order
    node.process_order(stitch_order)

    # After stitch: 3 action_states total
    assert len(node._current_state.action_states) == 3
    action_map = {a.action_id: a for a in node._current_state.action_states}

    # action1 (node1) preserved as FINISHED
    assert action_map[action1.action_id].action_status == CurrentAction.FINISHED

    # action2 (stitch node) preserved as RUNNING — not reset to WAITING
    assert action_map[action2.action_id].action_status == CurrentAction.RUNNING

    # action3 (new node) appended as WAITING
    assert action_map[action3.action_id].action_status == CurrentAction.WAITING


def test_stitch_no_actions_on_stitch_node(
    mocker,
    adapter_node,
    action_server_nav_to_node,
    action_server_process_vda_action,
    service_get_state,
    service_supported_actions,
):
    """
    Preserve action states when the stitch node has no actions.

    This guards against the ``[:-0] == [:0] == []`` slicing bug.
    """
    node = VDA5050Controller()
    node.logger.set_level(LoggingSeverity.DEBUG)

    order_id = str(uuid4())

    node_a_action = Action(
        action_type="honk", action_id="node_a_act_1",
        action_description="Honk at node A", blocking_type="NONE",
    )
    node_c_action = Action(
        action_type="dock", action_id="node_c_act_1",
        action_description="Dock at C", blocking_type="NONE",
    )

    # Base order: A(0 w/ action) --> B(2, stitch node) --> C(4 unreleased)
    # The horizon keeps has_current_order() True after reaching B.
    base_order = Order(
        header_id=0, timestamp=get_vda5050_ts(), version="1.1.1",
        manufacturer="MANUFACTURER", serial_number="SERIAL_NUMBER",
        order_id=order_id, order_update_id=0,
        nodes=[
            Node(node_id="A", sequence_id=0, released=True,
                 node_position=NodePosition(
                     x=0.0, y=0.0, theta=0.0, map_id="map"),
                 actions=[node_a_action]),
            Node(node_id="B", sequence_id=2, released=True,
                 node_position=NodePosition(x=1.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="C", sequence_id=4, released=False,
                 node_position=NodePosition(x=2.0, y=0.0, theta=0.0, map_id="map")),
        ],
        edges=[
            Edge(edge_id="e1", sequence_id=1, released=True,
                 start_node_id="A", end_node_id="B"),
            Edge(edge_id="e2", sequence_id=3, released=False,
                 start_node_id="B", end_node_id="C"),
        ],
    )

    # Stitch order: B(2, no actions) --> C(4, node_c_action)
    stitch_order = Order(
        header_id=0, timestamp=get_vda5050_ts(), version="1.1.1",
        manufacturer="MANUFACTURER", serial_number="SERIAL_NUMBER",
        order_id=order_id, order_update_id=1,
        nodes=[
            Node(node_id="B", sequence_id=2, released=True,
                 node_position=NodePosition(x=1.0, y=0.0, theta=0.0, map_id="map")),
            Node(node_id="C", sequence_id=4, released=True,
                 node_position=NodePosition(
                     x=2.0, y=0.0, theta=0.0, map_id="map"),
                 actions=[node_c_action]),
        ],
        edges=[
            Edge(edge_id="e2", sequence_id=3, released=True,
                 start_node_id="B", end_node_id="C"),
        ],
    )

    # Accept base and navigate to node B
    node.process_order(base_order)
    rclpy.spin_once(node)
    rclpy.spin_once(adapter_node)
    node._process_last_edge_node()

    # Mark the node action as FINISHED
    node._update_action_status(node_a_action.action_id, CurrentAction.FINISHED)

    # Pre-stitch: 1 action_state (node_a_action, FINISHED)
    assert len(node._current_state.action_states) == 1
    assert node._current_state.action_states[0].action_id == node_a_action.action_id
    assert node._current_state.action_states[0].action_status == CurrentAction.FINISHED

    # Process stitch — stitch node B has 0 actions
    node.process_order(stitch_order)

    # Existing action_states must be fully preserved (not wiped)
    # and the new node C's action appended
    assert len(node._current_state.action_states) == 2
    action_map = {a.action_id: a for a in node._current_state.action_states}
    assert action_map[node_a_action.action_id].action_status == CurrentAction.FINISHED
    assert action_map[node_c_action.action_id].action_status == CurrentAction.WAITING
