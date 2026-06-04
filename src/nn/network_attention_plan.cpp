/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_attention_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace MetalFish {
namespace NN {
namespace {

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.ends_with(suffix);
}

const NetworkResolvedTensorRef &
RequireTensorSuffix(const NetworkResolvedExecutionStep &step,
                    std::string_view suffix) {
  for (const auto &tensor : step.tensors) {
    if (EndsWith(tensor.name, suffix))
      return tensor;
  }
  throw std::runtime_error("attention plan is missing tensor " +
                           std::string(suffix) + " in " + step.name);
}

const NetworkResolvedExecutionStep *
FindStep(const NetworkResolvedExecutionPlan &execution_plan,
         std::string_view name) {
  for (const auto &step : execution_plan.steps) {
    if (step.name == name)
      return &step;
  }
  return nullptr;
}

void RequireMatrix(const NetworkResolvedTensorRef &tensor,
                   std::string_view role) {
  if (tensor.dims.size() != 2 || tensor.dims[0] == 0 || tensor.dims[1] == 0) {
    throw std::runtime_error("attention " + std::string(role) +
                             " tensor has invalid shape: " + tensor.name);
  }
}

void RequireVector(const NetworkResolvedTensorRef &tensor,
                   std::string_view role) {
  if (tensor.dims.size() != 1 || tensor.dims[0] == 0) {
    throw std::runtime_error("attention " + std::string(role) +
                             " tensor has invalid shape: " + tensor.name);
  }
}

void RequireBiasWidth(const NetworkResolvedTensorRef &bias, int width,
                      std::string_view role) {
  RequireVector(bias, role);
  if (bias.elements != static_cast<std::size_t>(width)) {
    throw std::runtime_error("attention " + std::string(role) +
                             " tensor width mismatch: " + bias.name);
  }
}

void ResolveSmolgenPlan(const NetworkResolvedExecutionPlan &execution_plan,
                        const NetworkResolvedExecutionStep &attention,
                        int input_width, int heads, SmolgenStagePlan &smolgen) {
  const auto *dense =
      FindStep(execution_plan, attention.name + ".smolgen.dense");
  if (!dense)
    return;

  const auto &compress = RequireTensorSuffix(*dense, ".compress");
  const auto &dense1_w = RequireTensorSuffix(*dense, ".dense1_w");
  const auto &dense1_b = RequireTensorSuffix(*dense, ".dense1_b");
  const auto &dense2_w = RequireTensorSuffix(*dense, ".dense2_w");
  const auto &dense2_b = RequireTensorSuffix(*dense, ".dense2_b");

  RequireMatrix(compress, "smolgen compress");
  RequireMatrix(dense1_w, "smolgen dense1 weight");
  RequireBiasWidth(dense1_b, static_cast<int>(dense1_w.dims[0]),
                   "smolgen dense1 bias");
  RequireMatrix(dense2_w, "smolgen dense2 weight");
  RequireBiasWidth(dense2_b, static_cast<int>(dense2_w.dims[0]),
                   "smolgen dense2 bias");

  if (compress.dims[1] != static_cast<std::uint32_t>(input_width)) {
    throw std::runtime_error("attention smolgen compress input width mismatch");
  }
  const int compressed_channels = static_cast<int>(compress.dims[0]);
  const int flattened_width = kAttentionSquares * compressed_channels;
  if (dense1_w.dims[1] != static_cast<std::uint32_t>(flattened_width)) {
    throw std::runtime_error("attention smolgen dense1 input width mismatch");
  }
  const int dense1_width = static_cast<int>(dense1_w.dims[0]);
  if (dense2_w.dims[1] != static_cast<std::uint32_t>(dense1_width)) {
    throw std::runtime_error("attention smolgen dense2 input width mismatch");
  }
  const int dense2_width = static_cast<int>(dense2_w.dims[0]);
  if (dense2_width % heads != 0) {
    throw std::runtime_error(
        "attention smolgen dense2 width is not divisible by heads");
  }

  const auto *norm = FindStep(execution_plan, attention.name + ".smolgen.norm");
  if (!norm)
    throw std::runtime_error("attention smolgen is missing layer norms");
  RequireBiasWidth(RequireTensorSuffix(*norm, ".ln1_gammas"), dense1_width,
                   "smolgen ln1 gamma");
  RequireBiasWidth(RequireTensorSuffix(*norm, ".ln1_betas"), dense1_width,
                   "smolgen ln1 beta");
  RequireBiasWidth(RequireTensorSuffix(*norm, ".ln2_gammas"), dense2_width,
                   "smolgen ln2 gamma");
  RequireBiasWidth(RequireTensorSuffix(*norm, ".ln2_betas"), dense2_width,
                   "smolgen ln2 beta");

  smolgen.present = true;
  smolgen.compressed_channels = compressed_channels;
  smolgen.dense1_width = dense1_width;
  smolgen.dense2_width = dense2_width;
  smolgen.dense2_width_per_head = dense2_width / heads;

  const auto *positional = FindGlobalPositionalEncodingStep(execution_plan);
  if (!positional)
    return;

  const auto &global = RequireTensorSuffix(*positional, "body.smolgen_w");
  RequireMatrix(global, "global smolgen");
  if (global.dims[0] != kAttentionSquares * kAttentionSquares ||
      global.dims[1] !=
          static_cast<std::uint32_t>(smolgen.dense2_width_per_head)) {
    throw std::runtime_error(
        "attention global smolgen tensor dimensions mismatch");
  }
  smolgen.has_global_positional_weights = true;
  smolgen.global_position_rows = static_cast<int>(global.dims[0]);
  smolgen.global_position_cols = static_cast<int>(global.dims[1]);
}

} // namespace

const NetworkResolvedExecutionStep *FindGlobalPositionalEncodingStep(
    const NetworkResolvedExecutionPlan &execution_plan) {
  for (const auto &step : execution_plan.steps) {
    if (step.kind == NetworkExecutionOpKind::PositionalEncoding)
      return &step;
  }
  return nullptr;
}

AttentionStagePlan
ResolveAttentionStagePlan(const NetworkResolvedExecutionPlan &execution_plan,
                          std::size_t attention_step_index, int head_count) {
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("attention plan step index is out of range");
  if (head_count <= 0)
    throw std::runtime_error("attention plan head count is invalid");

  const auto &attention = execution_plan.steps[attention_step_index];
  if (attention.kind != NetworkExecutionOpKind::Attention) {
    throw std::runtime_error("attention plan step is not attention: " +
                             attention.name);
  }

  const auto &q_w = RequireTensorSuffix(attention, ".q_w");
  const auto &q_b = RequireTensorSuffix(attention, ".q_b");
  const auto &k_w = RequireTensorSuffix(attention, ".k_w");
  const auto &k_b = RequireTensorSuffix(attention, ".k_b");
  const auto &v_w = RequireTensorSuffix(attention, ".v_w");
  const auto &v_b = RequireTensorSuffix(attention, ".v_b");
  const auto &dense_w = RequireTensorSuffix(attention, ".dense_w");
  const auto &dense_b = RequireTensorSuffix(attention, ".dense_b");

  RequireMatrix(q_w, "query weight");
  RequireMatrix(k_w, "key weight");
  RequireMatrix(v_w, "value weight");
  RequireMatrix(dense_w, "output projection weight");

  const int qkv_width = static_cast<int>(q_w.dims[0]);
  const int input_width = static_cast<int>(q_w.dims[1]);
  if (qkv_width <= 0 || input_width <= 0 || k_w.dims != q_w.dims ||
      v_w.dims != q_w.dims) {
    throw std::runtime_error("attention Q/K/V dimensions mismatch");
  }
  RequireBiasWidth(q_b, qkv_width, "query bias");
  RequireBiasWidth(k_b, qkv_width, "key bias");
  RequireBiasWidth(v_b, qkv_width, "value bias");

  if (qkv_width % head_count != 0)
    throw std::runtime_error("attention Q/K/V width is not divisible by heads");
  if (dense_w.dims[1] != static_cast<std::uint32_t>(qkv_width)) {
    throw std::runtime_error(
        "attention output projection input width mismatch");
  }
  const int output_width = static_cast<int>(dense_w.dims[0]);
  RequireBiasWidth(dense_b, output_width, "output projection bias");
  if (output_width != input_width) {
    throw std::runtime_error(
        "attention output projection must return model width");
  }

  AttentionStagePlan plan;
  plan.name = attention.name;
  plan.heads = head_count;
  plan.input_width = input_width;
  plan.qkv_width = qkv_width;
  plan.head_depth = qkv_width / head_count;
  plan.output_width = output_width;
  ResolveSmolgenPlan(execution_plan, attention, input_width, head_count,
                     plan.smolgen);
  return plan;
}

} // namespace NN
} // namespace MetalFish
