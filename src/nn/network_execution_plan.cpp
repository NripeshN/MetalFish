/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_execution_plan.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace MetalFish {
namespace NN {
namespace {

void AddError(NetworkExecutionValidation &validation, std::string error) {
  validation.errors.push_back(std::move(error));
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

bool HasTensorWithPrefix(const NetworkWeightInventory &inventory,
                         const std::string &prefix) {
  for (const auto &tensor : inventory.tensors) {
    if (StartsWith(tensor.name, prefix))
      return true;
  }
  return false;
}

void AddStep(NetworkExecutionPlan &plan, NetworkExecutionOpKind kind,
             std::string name, std::vector<std::string> tensors) {
  plan.steps.push_back(
      NetworkExecutionStep{kind, std::move(name), std::move(tensors)});
}

void AddStepIfAny(NetworkExecutionPlan &plan,
                  const NetworkWeightInventory &inventory,
                  NetworkExecutionOpKind kind, std::string name,
                  std::initializer_list<const char *> tensor_names) {
  std::vector<std::string> present;
  for (const char *tensor_name : tensor_names) {
    if (inventory.Contains(tensor_name))
      present.emplace_back(tensor_name);
  }
  if (!present.empty())
    AddStep(plan, kind, std::move(name), std::move(present));
}

void AddStepIfAny(NetworkExecutionPlan &plan,
                  const NetworkWeightInventory &inventory,
                  NetworkExecutionOpKind kind, const std::string &name,
                  const std::vector<std::string> &tensor_names) {
  std::vector<std::string> present;
  for (const auto &tensor_name : tensor_names) {
    if (inventory.Contains(tensor_name))
      present.push_back(tensor_name);
  }
  if (!present.empty())
    AddStep(plan, kind, name, std::move(present));
}

std::vector<std::string> WithPrefix(
    const std::string &prefix, std::initializer_list<const char *> suffixes) {
  std::vector<std::string> names;
  names.reserve(suffixes.size());
  for (const char *suffix : suffixes)
    names.push_back(prefix + suffix);
  return names;
}

void AddConvBlock(NetworkExecutionPlan &plan,
                  const NetworkWeightInventory &inventory,
                  const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Convolution, name,
               WithPrefix(prefix, {".weights", ".biases"}));
}

void AddLayerNorm(NetworkExecutionPlan &plan,
                  const NetworkWeightInventory &inventory,
                  const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::LayerNorm, name,
               WithPrefix(prefix, {"_gammas", "_betas"}));
}

void AddDense(NetworkExecutionPlan &plan, const NetworkWeightInventory &inventory,
              const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Dense, name,
               WithPrefix(prefix, {"_w", "_b"}));
}

void AddFFN(NetworkExecutionPlan &plan, const NetworkWeightInventory &inventory,
            const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::FeedForward, name,
               WithPrefix(prefix, {".dense1_w", ".dense1_b", ".dense2_w",
                                   ".dense2_b"}));
}

void AddSmolgen(NetworkExecutionPlan &plan,
                const NetworkWeightInventory &inventory,
                const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Dense, name + ".dense",
               WithPrefix(prefix, {".compress", ".dense1_w", ".dense1_b",
                                   ".dense2_w", ".dense2_b"}));
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::LayerNorm,
               name + ".norm",
               WithPrefix(prefix, {".ln1_gammas", ".ln1_betas",
                                   ".ln2_gammas", ".ln2_betas"}));
}

void AddMHA(NetworkExecutionPlan &plan, const NetworkWeightInventory &inventory,
            const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Attention, name,
               WithPrefix(prefix, {".q_w", ".q_b", ".k_w", ".k_b", ".v_w",
                                   ".v_b", ".dense_w", ".dense_b"}));
  AddSmolgen(plan, inventory, prefix + ".smolgen", name + ".smolgen");
}

void AddEncoder(NetworkExecutionPlan &plan,
                const NetworkWeightInventory &inventory,
                const std::string &prefix, const std::string &name) {
  AddMHA(plan, inventory, prefix + ".mha", name + ".mha");
  AddLayerNorm(plan, inventory, prefix + ".ln1", name + ".ln1");
  AddFFN(plan, inventory, prefix + ".ffn", name + ".ffn");
  AddLayerNorm(plan, inventory, prefix + ".ln2", name + ".ln2");
}

void AddEncoderSeries(NetworkExecutionPlan &plan,
                      const NetworkWeightInventory &inventory,
                      const std::string &prefix, const std::string &name) {
  for (std::size_t i = 0;; ++i) {
    const std::string layer_prefix = prefix + "." + std::to_string(i);
    if (!HasTensorWithPrefix(inventory, layer_prefix + "."))
      break;
    AddEncoder(plan, inventory, layer_prefix, name + "." + std::to_string(i));
  }
}

void AddBody(NetworkExecutionPlan &plan,
             const NetworkWeightInventory &inventory) {
  AddConvBlock(plan, inventory, "body.input", "body.input");
  AddDense(plan, inventory, "body.ip_emb_preproc",
           "body.input_embedding_preprocess");
  AddDense(plan, inventory, "body.ip_emb", "body.input_embedding");
  AddLayerNorm(plan, inventory, "body.ip_emb_ln", "body.input_embedding_norm");
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Gate,
               "body.input_embedding_gates",
               {"body.ip_mult_gate", "body.ip_add_gate"});
  AddFFN(plan, inventory, "body.ip_emb_ffn", "body.input_embedding_ffn");
  AddLayerNorm(plan, inventory, "body.ip_emb_ffn_ln",
               "body.input_embedding_ffn_norm");
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::PositionalEncoding,
               "body.smolgen_positional", {"body.smolgen_w"});
  AddEncoderSeries(plan, inventory, "body.encoder", "body.encoder");

  for (std::size_t i = 0;; ++i) {
    const std::string prefix = "body.residual." + std::to_string(i);
    if (!HasTensorWithPrefix(inventory, prefix + "."))
      break;
    const std::string name = "body.residual." + std::to_string(i);
    AddConvBlock(plan, inventory, prefix + ".conv1", name + ".conv1");
    AddConvBlock(plan, inventory, prefix + ".conv2", name + ".conv2");
    AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Dense, name + ".se",
                 WithPrefix(prefix + ".se", {".w1", ".b1", ".w2", ".b2"}));
  }
}

void AddMovesLeft(NetworkExecutionPlan &plan,
                  const NetworkWeightInventory &inventory) {
  AddConvBlock(plan, inventory, "moves_left", "moves_left.conv");
  AddDense(plan, inventory, "moves_left.ip_mov", "moves_left.dense0");
  AddDense(plan, inventory, "moves_left.ip1_mov", "moves_left.dense1");
  AddDense(plan, inventory, "moves_left.ip2_mov", "moves_left.output");
}

void AddPolicy(NetworkExecutionPlan &plan,
               const NetworkWeightInventory &inventory,
               const std::string &policy_head) {
  const std::string prefix = "policy." + policy_head;
  const std::string name = "policy." + policy_head;
  AddConvBlock(plan, inventory, prefix + ".policy1", name + ".policy1");
  AddConvBlock(plan, inventory, prefix + ".policy", name + ".policy");
  AddEncoderSeries(plan, inventory, prefix + ".encoder", name + ".encoder");
  AddDense(plan, inventory, prefix + ".ip_pol", name + ".output");
  AddDense(plan, inventory, prefix + ".ip2_pol", name + ".dense2");
  AddDense(plan, inventory, prefix + ".ip3_pol", name + ".dense3");
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::PolicyMap,
               name + ".policy_map",
               std::vector<std::string>{prefix + ".ip4_pol_w"});
}

void AddValue(NetworkExecutionPlan &plan,
              const NetworkWeightInventory &inventory,
              const std::string &value_head) {
  const std::string prefix = "value." + value_head;
  const std::string name = "value." + value_head;
  AddConvBlock(plan, inventory, prefix + ".value", name + ".value");
  AddDense(plan, inventory, prefix + ".ip_val", name + ".output");
  AddDense(plan, inventory, prefix + ".ip1_val", name + ".dense1");
  AddDense(plan, inventory, prefix + ".ip2_val", name + ".dense2");
  AddDense(plan, inventory, prefix + ".ip_val_err", name + ".error");
}

bool HasFlatShape(const NetworkResolvedTensorRef &tensor) {
  return tensor.dims.empty() ||
         (tensor.dims.size() == 1 && tensor.dims[0] == tensor.elements);
}

void SetFlatShape(NetworkResolvedTensorRef &tensor,
                  std::vector<std::uint32_t> dims) {
  if (!HasFlatShape(tensor) || dims.empty())
    return;
  std::size_t elements = 1;
  for (std::uint32_t dim : dims)
    elements *= dim;
  if (elements == tensor.elements)
    tensor.dims = std::move(dims);
}

NetworkResolvedTensorRef *
FindTensorSuffix(NetworkResolvedExecutionStep &step, std::string_view suffix) {
  for (auto &tensor : step.tensors) {
    if (EndsWith(tensor.name, suffix))
      return &tensor;
  }
  return nullptr;
}

const NetworkResolvedTensorRef *
FindTensorSuffix(const NetworkResolvedExecutionStep &step,
                 std::string_view suffix) {
  for (const auto &tensor : step.tensors) {
    if (EndsWith(tensor.name, suffix))
      return &tensor;
  }
  return nullptr;
}

NetworkResolvedExecutionStep *
FindResolvedStep(NetworkResolvedExecutionPlan &plan, std::string_view name) {
  for (auto &step : plan.steps) {
    if (step.name == name)
      return &step;
  }
  return nullptr;
}

const NetworkResolvedExecutionStep *
FindResolvedStep(const NetworkResolvedExecutionPlan &plan,
                 std::string_view name) {
  for (const auto &step : plan.steps) {
    if (step.name == name)
      return &step;
  }
  return nullptr;
}

void InferMatrixFromBias(NetworkResolvedTensorRef *weight,
                         NetworkResolvedTensorRef *bias) {
  if (!weight || !bias || bias->elements == 0 ||
      weight->elements % bias->elements != 0) {
    return;
  }
  SetFlatShape(*weight,
               {static_cast<std::uint32_t>(bias->elements),
                static_cast<std::uint32_t>(weight->elements / bias->elements)});
  SetFlatShape(*bias, {static_cast<std::uint32_t>(bias->elements)});
}

void InferDenseStepShapes(NetworkResolvedExecutionStep &step) {
  if (step.kind != NetworkExecutionOpKind::Dense)
    return;
  if (EndsWith(step.name, ".smolgen.dense"))
    return;

  InferMatrixFromBias(FindTensorSuffix(step, "_w"), FindTensorSuffix(step, "_b"));
}

void InferFeedForwardStepShapes(NetworkResolvedExecutionStep &step) {
  if (step.kind != NetworkExecutionOpKind::FeedForward)
    return;

  auto *dense1_w = FindTensorSuffix(step, ".dense1_w");
  auto *dense1_b = FindTensorSuffix(step, ".dense1_b");
  auto *dense2_w = FindTensorSuffix(step, ".dense2_w");
  auto *dense2_b = FindTensorSuffix(step, ".dense2_b");
  InferMatrixFromBias(dense1_w, dense1_b);
  if (dense1_b && dense2_w && dense2_b &&
      dense2_w->elements == dense2_b->elements * dense1_b->elements) {
    SetFlatShape(*dense2_w,
                 {static_cast<std::uint32_t>(dense2_b->elements),
                  static_cast<std::uint32_t>(dense1_b->elements)});
    SetFlatShape(*dense2_b, {static_cast<std::uint32_t>(dense2_b->elements)});
  } else {
    InferMatrixFromBias(dense2_w, dense2_b);
  }
}

void InferAttentionStepShapes(NetworkResolvedExecutionStep &step) {
  if (step.kind != NetworkExecutionOpKind::Attention)
    return;

  auto *q_w = FindTensorSuffix(step, ".q_w");
  auto *q_b = FindTensorSuffix(step, ".q_b");
  auto *k_w = FindTensorSuffix(step, ".k_w");
  auto *k_b = FindTensorSuffix(step, ".k_b");
  auto *v_w = FindTensorSuffix(step, ".v_w");
  auto *v_b = FindTensorSuffix(step, ".v_b");
  auto *dense_w = FindTensorSuffix(step, ".dense_w");
  auto *dense_b = FindTensorSuffix(step, ".dense_b");
  InferMatrixFromBias(q_w, q_b);
  InferMatrixFromBias(k_w, k_b);
  InferMatrixFromBias(v_w, v_b);
  if (q_b && dense_w && dense_b &&
      dense_w->elements == dense_b->elements * q_b->elements) {
    SetFlatShape(*dense_w,
                 {static_cast<std::uint32_t>(dense_b->elements),
                  static_cast<std::uint32_t>(q_b->elements)});
    SetFlatShape(*dense_b, {static_cast<std::uint32_t>(dense_b->elements)});
  } else {
    InferMatrixFromBias(dense_w, dense_b);
  }
}

void InferPolicyMapStepShapes(NetworkResolvedExecutionStep &step) {
  if (step.kind != NetworkExecutionOpKind::PolicyMap || step.tensors.empty())
    return;
  auto &promotion_weights = step.tensors[0];
  if (promotion_weights.elements % 4 == 0) {
    SetFlatShape(promotion_weights,
                 {4, static_cast<std::uint32_t>(promotion_weights.elements / 4)});
  }
}

void InferSmolgenStepShapes(NetworkResolvedExecutionPlan &plan,
                            NetworkResolvedExecutionStep &step) {
  if (step.kind != NetworkExecutionOpKind::Dense ||
      !EndsWith(step.name, ".smolgen.dense")) {
    return;
  }

  const std::string attention_name =
      step.name.substr(0, step.name.size() - std::string(".smolgen.dense").size());
  const auto *attention = FindResolvedStep(plan, attention_name);
  if (!attention)
    return;
  const auto *q_w = FindTensorSuffix(*attention, ".q_w");
  if (!q_w || q_w->dims.size() != 2 || q_w->dims[1] == 0)
    return;

  auto *compress = FindTensorSuffix(step, ".compress");
  auto *dense1_w = FindTensorSuffix(step, ".dense1_w");
  auto *dense1_b = FindTensorSuffix(step, ".dense1_b");
  auto *dense2_w = FindTensorSuffix(step, ".dense2_w");
  auto *dense2_b = FindTensorSuffix(step, ".dense2_b");
  if (!compress || !dense1_w || !dense1_b || !dense2_w || !dense2_b)
    return;

  const std::uint32_t parent_width = q_w->dims[1];
  if (compress->elements % parent_width != 0)
    return;
  const std::uint32_t compressed_channels =
      static_cast<std::uint32_t>(compress->elements / parent_width);
  SetFlatShape(*compress, {compressed_channels, parent_width});

  const std::uint32_t flattened_width =
      static_cast<std::uint32_t>(kPackedInputSquareCount) * compressed_channels;
  if (dense1_w->elements == dense1_b->elements * flattened_width) {
    SetFlatShape(*dense1_w,
                 {static_cast<std::uint32_t>(dense1_b->elements),
                  flattened_width});
    SetFlatShape(*dense1_b, {static_cast<std::uint32_t>(dense1_b->elements)});
  }
  if (dense2_w->elements == dense2_b->elements * dense1_b->elements) {
    SetFlatShape(*dense2_w,
                 {static_cast<std::uint32_t>(dense2_b->elements),
                  static_cast<std::uint32_t>(dense1_b->elements)});
    SetFlatShape(*dense2_b, {static_cast<std::uint32_t>(dense2_b->elements)});
  }
}

void InferPositionalEncodingShapes(NetworkResolvedExecutionPlan &plan) {
  NetworkResolvedTensorRef *global = nullptr;
  for (auto &step : plan.steps) {
    if (step.kind != NetworkExecutionOpKind::PositionalEncoding)
      continue;
    global = FindTensorSuffix(step, "body.smolgen_w");
    if (global)
      break;
  }
  if (!global || global->elements % (kPackedInputSquareCount *
                                     kPackedInputSquareCount) != 0) {
    return;
  }
  SetFlatShape(*global,
               {static_cast<std::uint32_t>(kPackedInputSquareCount *
                                           kPackedInputSquareCount),
                static_cast<std::uint32_t>(
                    global->elements /
                    (kPackedInputSquareCount * kPackedInputSquareCount))});
}

void InferResolvedExecutionTensorShapes(NetworkResolvedExecutionPlan &plan) {
  for (auto &step : plan.steps) {
    InferDenseStepShapes(step);
    InferFeedForwardStepShapes(step);
    InferAttentionStepShapes(step);
    InferPolicyMapStepShapes(step);
  }
  for (auto &step : plan.steps)
    InferSmolgenStepShapes(plan, step);
  InferPositionalEncodingShapes(plan);
}

} // namespace

std::string NetworkExecutionValidation::Summary() const {
  if (errors.empty())
    return "ok";
  std::ostringstream out;
  for (std::size_t i = 0; i < errors.size(); ++i) {
    if (i > 0)
      out << "; ";
    out << errors[i];
  }
  return out.str();
}

std::string NetworkResolvedTensorRef::ShapeString() const {
  if (dims.empty())
    return "flat";
  std::ostringstream out;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i > 0)
      out << "x";
    out << dims[i];
  }
  return out.str();
}

std::size_t NetworkResolvedExecutionStep::ParameterElements() const {
  std::size_t total = 0;
  for (const auto &tensor : tensors)
    total += tensor.elements;
  return total;
}

std::size_t NetworkResolvedExecutionStep::ParameterBytes() const {
  return ParameterElements() * sizeof(float);
}

bool NetworkResolvedExecutionStep::HasTensorKind(
    NetworkWeightTensorKind kind) const {
  for (const auto &tensor : tensors) {
    if (tensor.kind == kind)
      return true;
  }
  return false;
}

bool NetworkResolvedExecutionPlan::ContainsStep(std::string_view name) const {
  for (const auto &step : steps) {
    if (step.name == name)
      return true;
  }
  return false;
}

bool NetworkResolvedExecutionPlan::ReferencesTensor(
    std::string_view name) const {
  for (const auto &step : steps) {
    for (const auto &tensor : step.tensors) {
      if (tensor.name == name)
        return true;
    }
  }
  return false;
}

std::size_t NetworkResolvedExecutionPlan::TensorReferenceCount() const {
  std::size_t count = 0;
  for (const auto &step : steps)
    count += step.tensors.size();
  return count;
}

std::size_t NetworkResolvedExecutionPlan::TotalParameterElements() const {
  std::size_t total = 0;
  for (const auto &step : steps)
    total += step.ParameterElements();
  return total;
}

std::size_t NetworkResolvedExecutionPlan::TotalParameterBytes() const {
  return TotalParameterElements() * sizeof(float);
}

std::size_t
NetworkResolvedExecutionPlan::StepCount(NetworkExecutionOpKind kind) const {
  std::size_t count = 0;
  for (const auto &step : steps) {
    if (step.kind == kind)
      ++count;
  }
  return count;
}

std::string NetworkResolvedExecutionPlan::Summary() const {
  std::ostringstream out;
  out << steps.size() << " resolved execution steps, "
      << TensorReferenceCount() << " tensor refs, "
      << TotalParameterElements() << " parameter floats, policy_head="
      << policy_head << ", value_head=" << value_head;
  return out.str();
}

bool NetworkExecutionPlan::ContainsStep(std::string_view name) const {
  for (const auto &step : steps) {
    if (step.name == name)
      return true;
  }
  return false;
}

bool NetworkExecutionPlan::ReferencesTensor(std::string_view name) const {
  for (const auto &step : steps) {
    for (const auto &tensor : step.tensors) {
      if (tensor == name)
        return true;
    }
  }
  return false;
}

std::size_t NetworkExecutionPlan::TensorReferenceCount() const {
  std::size_t count = 0;
  for (const auto &step : steps)
    count += step.tensors.size();
  return count;
}

std::string NetworkExecutionPlan::Summary() const {
  std::ostringstream out;
  out << steps.size() << " execution steps, " << TensorReferenceCount()
      << " tensor refs, policy_head=" << policy_head
      << ", value_head=" << value_head;
  return out.str();
}

NetworkExecutionValidation NetworkExecutionPlan::ValidateAgainstInventory(
    const NetworkWeightInventory &inventory) const {
  NetworkExecutionValidation validation;

  std::string shape_error;
  if (!inventory.AllShapesMatchElements(&shape_error))
    AddError(validation, shape_error);

  std::set<std::string> referenced;
  for (const auto &step : steps) {
    for (const auto &tensor : step.tensors) {
      if (!inventory.Contains(tensor)) {
        AddError(validation,
                 "execution step references missing tensor: " + step.name +
                     " -> " + tensor);
      }
      referenced.insert(tensor);
    }
  }

  for (const auto &tensor : inventory.tensors) {
    if (referenced.find(tensor.name) == referenced.end()) {
      AddError(validation, "inventory tensor is not scheduled: " + tensor.name);
    }
  }

  return validation;
}

std::string NetworkExecutionOpKindName(NetworkExecutionOpKind kind) {
  switch (kind) {
  case NetworkExecutionOpKind::InputPack:
    return "input_pack";
  case NetworkExecutionOpKind::Convolution:
    return "convolution";
  case NetworkExecutionOpKind::Dense:
    return "dense";
  case NetworkExecutionOpKind::LayerNorm:
    return "layer_norm";
  case NetworkExecutionOpKind::Gate:
    return "gate";
  case NetworkExecutionOpKind::PositionalEncoding:
    return "positional_encoding";
  case NetworkExecutionOpKind::Attention:
    return "attention";
  case NetworkExecutionOpKind::FeedForward:
    return "feed_forward";
  case NetworkExecutionOpKind::PolicyMap:
    return "policy_map";
  case NetworkExecutionOpKind::OutputDecode:
    return "output_decode";
  }
  return "unknown";
}

NetworkExecutionPlan CreateNetworkExecutionPlan(
    const NetworkFormatDescriptor &format, const NetworkTensorPlan &tensors,
    const std::string &policy_head, const std::string &value_head,
    const NetworkWeightInventory &inventory) {
  NetworkExecutionPlan plan;
  plan.format = format;
  plan.tensors = tensors;
  plan.policy_head = policy_head;
  plan.value_head = value_head;

  AddStep(plan, NetworkExecutionOpKind::InputPack, "input.pack", {});
  AddBody(plan, inventory);
  if (tensors.moves_left)
    AddMovesLeft(plan, inventory);
  AddPolicy(plan, inventory, policy_head);
  AddValue(plan, inventory, value_head);
  AddStep(plan, NetworkExecutionOpKind::OutputDecode, "output.decode", {});

  return plan;
}

NetworkResolvedExecutionPlan ResolveNetworkExecutionPlan(
    const NetworkExecutionPlan &plan, const NetworkWeightInventory &inventory) {
  const auto validation = plan.ValidateAgainstInventory(inventory);
  if (!validation.ok()) {
    throw std::runtime_error("cannot resolve invalid NN execution plan: " +
                             validation.Summary());
  }

  NetworkResolvedExecutionPlan resolved;
  resolved.format = plan.format;
  resolved.tensors = plan.tensors;
  resolved.policy_head = plan.policy_head;
  resolved.value_head = plan.value_head;
  resolved.steps.reserve(plan.steps.size());

  for (const auto &step : plan.steps) {
    NetworkResolvedExecutionStep resolved_step;
    resolved_step.kind = step.kind;
    resolved_step.name = step.name;
    resolved_step.tensors.reserve(step.tensors.size());
    for (const auto &tensor_name : step.tensors) {
      bool found = false;
      for (std::size_t i = 0; i < inventory.tensors.size(); ++i) {
        const auto &tensor = inventory.tensors[i];
        if (tensor.name != tensor_name)
          continue;
        resolved_step.tensors.push_back(NetworkResolvedTensorRef{
            i, tensor.name, tensor.elements, tensor.dims, tensor.kind});
        found = true;
        break;
      }
      if (!found) {
        throw std::runtime_error("cannot resolve missing NN tensor: " +
                                 tensor_name);
      }
    }
    resolved.steps.push_back(std::move(resolved_step));
  }

  InferResolvedExecutionTensorShapes(resolved);
  return resolved;
}

} // namespace NN
} // namespace MetalFish
