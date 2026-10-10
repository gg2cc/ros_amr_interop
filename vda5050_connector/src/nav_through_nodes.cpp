// BSD 3-Clause License
//
// Copyright (c) 2022 InOrbit, Inc.
// Copyright (c) 2022 Clearpath Robotics, Inc.
// Copyright (c) 2026 Quantillion Technologies
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the InOrbit, Inc. nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "vda5050_connector/nav_through_nodes.hpp"

#include <rclcpp/logging.hpp>

void adapter::NavThroughNodes::setupExtendNavigationService()
{
  std::string manufacturer_name = node_->get_parameter("manufacturer_name").as_string();
  std::string base_interface_name =
    std::string(node_->get_namespace()) + "/" + manufacturer_name + "/" + robot_name_ + "/";
  extend_navigation_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  extend_navigation_srv_ = node_->create_service<ExtendNavigation>(
    base_interface_name + "adapter/extend_navigation",
    std::bind(
      &NavThroughNodes::extendNavigationCallback, this, std::placeholders::_1,
      std::placeholders::_2),
    rmw_qos_profile_services_default, extend_navigation_callback_group_);
}

void adapter::NavThroughNodes::extendNavigationCallback(
  const std::shared_ptr<ExtendNavigation::Request> request,
  std::shared_ptr<ExtendNavigation::Response> response)
{
  std::lock_guard<std::mutex> extension_lock(extension_mutex_);

  if (request->edges.empty() || request->nodes.size() < 2) {
    response->success = false;
    response->message = "Need at least 1 edge and 2 nodes (stitch + target)";
    return;
  }

  if (request->edges.size() != request->nodes.size() - 1) {
    response->success = false;
    response->message = "Size mismatch: edges must equal nodes - 1";
    return;
  }

  size_t old_edge_count = 0;
  size_t old_node_count = 0;
  {
    std::shared_lock lock(navigation_mutex_);

    if (!goal_handle_ || !goal_handle_->is_active()) {
      response->success = false;
      response->message = "No active navigation goal";
      return;
    }

    if (nodes_msg_.empty()) {
      response->success = false;
      response->message = "Cannot stitch: no existing navigation nodes available";
      RCLCPP_WARN(node_->get_logger(), "NavThroughNodes: %s", response->message.c_str());
      return;
    }

    // 先核对 stitch 引用点，避免底层接受后才发现 connector 快照不匹配。
    const auto & stitch_node = request->nodes[0];
    const auto & expected_node = nodes_msg_.back();
    if (
      stitch_node.node_id != expected_node.node_id ||
      stitch_node.sequence_id != expected_node.sequence_id) {
      response->success = false;
      response->message = "Stitch node mismatch: expected node_id='" + expected_node.node_id +
                          "' seq=" + std::to_string(expected_node.sequence_id) + ", got node_id='" +
                          stitch_node.node_id + "' seq=" + std::to_string(stitch_node.sequence_id);
      RCLCPP_WARN(node_->get_logger(), "NavThroughNodes: %s", response->message.c_str());
      return;
    }

    old_edge_count = edges_msg_.size();
    old_node_count = nodes_msg_.size();
  }

  std::string validation_message;
  if (!validateNavigationExtension(request->edges, request->nodes, validation_message)) {
    response->success = false;
    response->message =
      validation_message.empty() ? "Navigation extension rejected by handler" : validation_message;
    RCLCPP_WARN(node_->get_logger(), "NavThroughNodes: %s", response->message.c_str());
    return;
  }

  {
    std::unique_lock lock(navigation_mutex_);

    // 底层校验期间若 goal 或路径版本变化，不提交过期的 connector 快照。
    if (
      !goal_handle_ || !goal_handle_->is_active() || edges_msg_.size() != old_edge_count ||
      nodes_msg_.size() != old_node_count || nodes_msg_.empty() ||
      nodes_msg_.back().node_id != request->nodes[0].node_id ||
      nodes_msg_.back().sequence_id != request->nodes[0].sequence_id) {
      response->success = false;
      response->message = "Navigation changed while extension was being validated";
      return;
    }

    // Append only the new edges and target nodes (skip stitch node)
    edges_msg_.insert(edges_msg_.end(), request->edges.begin(), request->edges.end());
    nodes_msg_.insert(nodes_msg_.end(), request->nodes.begin() + 1, request->nodes.end());
  }

  onNavigationExtended(old_edge_count);

  size_t new_count = request->nodes.size() - 1;
  response->success = true;
  response->message = "Extended navigation with " + std::to_string(new_count) + " new nodes";

  RCLCPP_INFO(node_->get_logger(), "NavThroughNodes: %s", response->message.c_str());
}
