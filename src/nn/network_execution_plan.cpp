/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_execution_plan.h"

#include <set>
#include <sstream>
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

std::vector<std::string>
WithPrefix(const std::string &prefix,
           std::initializer_list<const char *> suffixes) {
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

void AddDense(NetworkExecutionPlan &plan,
              const NetworkWeightInventory &inventory,
              const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Dense, name,
               WithPrefix(prefix, {"_w", "_b"}));
}

void AddFFN(NetworkExecutionPlan &plan, const NetworkWeightInventory &inventory,
            const std::string &prefix, const std::string &name) {
  AddStepIfAny(
      plan, inventory, NetworkExecutionOpKind::FeedForward, name,
      WithPrefix(prefix, {".dense1_w", ".dense1_b", ".dense2_w", ".dense2_b"}));
}

void AddSmolgen(NetworkExecutionPlan &plan,
                const NetworkWeightInventory &inventory,
                const std::string &prefix, const std::string &name) {
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::Dense, name + ".dense",
               WithPrefix(prefix, {".compress", ".dense1_w", ".dense1_b",
                                   ".dense2_w", ".dense2_b"}));
  AddStepIfAny(plan, inventory, NetworkExecutionOpKind::LayerNorm,
               name + ".norm",
               WithPrefix(prefix, {".ln1_gammas", ".ln1_betas", ".ln2_gammas",
                                   ".ln2_betas"}));
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
        AddError(validation, "execution step references missing tensor: " +
                                 step.name + " -> " + tensor);
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

} // namespace NN
} // namespace MetalFish
