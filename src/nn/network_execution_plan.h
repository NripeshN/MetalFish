/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "network_format_types.h"
#include "network_tensor_plan.h"
#include "network_weight_inventory.h"

namespace MetalFish {
namespace NN {

enum class NetworkExecutionOpKind {
  InputPack,
  Convolution,
  Dense,
  LayerNorm,
  Gate,
  PositionalEncoding,
  Attention,
  FeedForward,
  PolicyMap,
  OutputDecode,
};

struct NetworkExecutionStep {
  NetworkExecutionOpKind kind = NetworkExecutionOpKind::Dense;
  std::string name;
  std::vector<std::string> tensors;
};

struct NetworkResolvedTensorRef {
  std::size_t inventory_index = 0;
  std::string name;
  std::size_t elements = 0;
  std::vector<std::uint32_t> dims;
  NetworkWeightTensorKind kind = NetworkWeightTensorKind::Generic;

  std::string ShapeString() const;
};

struct NetworkResolvedExecutionStep {
  NetworkExecutionOpKind kind = NetworkExecutionOpKind::Dense;
  std::string name;
  std::vector<NetworkResolvedTensorRef> tensors;

  std::size_t ParameterElements() const;
  std::size_t ParameterBytes() const;
  bool HasTensorKind(NetworkWeightTensorKind kind) const;
};

struct NetworkExecutionValidation {
  std::vector<std::string> errors;

  bool ok() const { return errors.empty(); }
  std::string Summary() const;
};

struct NetworkResolvedExecutionPlan {
  NetworkFormatDescriptor format;
  NetworkTensorPlan tensors;
  std::string policy_head;
  std::string value_head;
  std::vector<NetworkResolvedExecutionStep> steps;

  bool ContainsStep(std::string_view name) const;
  bool ReferencesTensor(std::string_view name) const;
  std::size_t TensorReferenceCount() const;
  std::size_t TotalParameterElements() const;
  std::size_t TotalParameterBytes() const;
  std::size_t StepCount(NetworkExecutionOpKind kind) const;
  std::string Summary() const;
};

struct NetworkExecutionPlan {
  NetworkFormatDescriptor format;
  NetworkTensorPlan tensors;
  std::string policy_head;
  std::string value_head;
  std::vector<NetworkExecutionStep> steps;

  bool ContainsStep(std::string_view name) const;
  bool ReferencesTensor(std::string_view name) const;
  std::size_t TensorReferenceCount() const;
  std::string Summary() const;
  NetworkExecutionValidation
  ValidateAgainstInventory(const NetworkWeightInventory &inventory) const;
};

struct NetworkPositionEncodingGeometry {
  int input_planes = 0;
  int input_squares = 0;
  int position_planes = 0;
  int position_width = 0;
  int dense_input_width = 0;
  int dense_output_width = 0;
  int concat_width = 0;
};

enum class NetworkPlanStageGroup {
  Other,
  Body,
  Policy,
  Value,
  MovesLeft,
};

struct NetworkFeedForwardStageWidths {
  int hidden = 0;
  int output = 0;
};

struct NetworkSqueezeExciteStageWidths {
  int hidden = 0;
  int output = 0;
};

std::string NetworkExecutionOpKindName(NetworkExecutionOpKind kind);

NetworkExecutionPlan CreateNetworkExecutionPlan(
    const NetworkFormatDescriptor &format, const NetworkTensorPlan &tensors,
    const std::string &policy_head, const std::string &value_head,
    const NetworkWeightInventory &inventory);

NetworkResolvedExecutionPlan
ResolveNetworkExecutionPlan(const NetworkExecutionPlan &plan,
                            const NetworkWeightInventory &inventory);

NetworkWeightInventory CreateResolvedNetworkWeightInventory(
    const NetworkWeightInventory &inventory,
    const NetworkResolvedExecutionPlan &resolved_plan);

NetworkPositionEncodingGeometry ResolveDynamicPositionEncodingGeometry(
    const NetworkResolvedExecutionPlan &plan,
    const NetworkResolvedExecutionStep &dense);

NetworkPositionEncodingGeometry ResolveStaticPositionEncodingGeometry(
    const NetworkResolvedExecutionPlan &plan,
    const NetworkResolvedExecutionStep &dense);

const NetworkResolvedTensorRef *
FindNetworkTensorSuffix(const NetworkResolvedExecutionStep &step,
                        std::string_view suffix);

std::string NetworkPlanStagePrefix(const NetworkResolvedExecutionPlan &plan,
                                   NetworkPlanStageGroup group);

NetworkPlanStageGroup
ClassifyNetworkPlanStage(const NetworkResolvedExecutionPlan &plan,
                         std::string_view stage_name);

bool IsNetworkValueErrorStage(std::string_view stage_name);

int NetworkDenseStageOutputWidth(const NetworkResolvedExecutionStep &dense);

int NetworkConvolutionStageOutputChannels(
    const NetworkResolvedExecutionStep &convolution);

int NetworkSqueezeExciteStageOutputWidth(
    const NetworkResolvedExecutionStep &se);

NetworkSqueezeExciteStageWidths
NetworkSqueezeExciteStageWidthsFor(const NetworkResolvedExecutionStep &se);

int NetworkLayerNormStageWidth(const NetworkResolvedExecutionStep &norm);

int NetworkGateStageWidth(const NetworkResolvedExecutionStep &gate);

NetworkFeedForwardStageWidths
NetworkFeedForwardStageWidthsFor(const NetworkResolvedExecutionStep &ffn);

int NetworkAttentionStageOutputWidth(
    const NetworkResolvedExecutionStep &attention);

int NetworkAttentionHeadCount(const NetworkResolvedExecutionPlan &plan,
                              std::string_view stage_name);

bool NetworkIsAttentionLayerNormStage(
    const NetworkResolvedExecutionPlan &plan, std::string_view stage_name);

bool NetworkStageUsesSquareRows(const NetworkResolvedExecutionPlan &plan,
                                std::string_view stage_name);

int NetworkDenseLikeRows(const NetworkResolvedExecutionPlan &plan,
                         std::string_view stage_name, int batch_size);

std::string NetworkDenseStageActivationName(
    const NetworkResolvedExecutionPlan &plan, std::string_view stage_name);

std::size_t
NetworkBodyEncoderLayerCount(const NetworkResolvedExecutionPlan &plan);

float NetworkFeedForwardResidualScale(
    const NetworkResolvedExecutionPlan &plan, std::string_view stage_name);

float NetworkFeedForwardLayerNormEpsilon(
    const NetworkResolvedExecutionPlan &plan, std::string_view stage_name);

float NetworkDenseLayerNormEpsilon(const NetworkResolvedExecutionPlan &plan,
                                   std::string_view stage_name);

float NetworkAttentionLayerNormEpsilon(const NetworkResolvedExecutionPlan &plan,
                                       std::string_view stage_name);

} // namespace NN
} // namespace MetalFish
