/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_weight_inventory.h"

#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace MetalFish {
namespace NN {
namespace {

void AddVector(NetworkWeightInventory &inventory, std::string name,
               const std::vector<float> &values) {
  if (values.empty())
    return;
  inventory.tensors.push_back(
      NetworkWeightTensorView{std::move(name), values.data(), values.size()});
}

void AddConvBlock(NetworkWeightInventory &inventory, const std::string &prefix,
                  const BaseWeights::ConvBlock &block) {
  AddVector(inventory, prefix + ".weights", block.weights);
  AddVector(inventory, prefix + ".biases", block.biases);
}

void AddSmolgen(NetworkWeightInventory &inventory, const std::string &prefix,
                const BaseWeights::Smolgen &smolgen) {
  AddVector(inventory, prefix + ".compress", smolgen.compress);
  AddVector(inventory, prefix + ".dense1_w", smolgen.dense1_w);
  AddVector(inventory, prefix + ".dense1_b", smolgen.dense1_b);
  AddVector(inventory, prefix + ".ln1_gammas", smolgen.ln1_gammas);
  AddVector(inventory, prefix + ".ln1_betas", smolgen.ln1_betas);
  AddVector(inventory, prefix + ".dense2_w", smolgen.dense2_w);
  AddVector(inventory, prefix + ".dense2_b", smolgen.dense2_b);
  AddVector(inventory, prefix + ".ln2_gammas", smolgen.ln2_gammas);
  AddVector(inventory, prefix + ".ln2_betas", smolgen.ln2_betas);
}

void AddMHA(NetworkWeightInventory &inventory, const std::string &prefix,
            const BaseWeights::MHA &mha) {
  AddVector(inventory, prefix + ".q_w", mha.q_w);
  AddVector(inventory, prefix + ".q_b", mha.q_b);
  AddVector(inventory, prefix + ".k_w", mha.k_w);
  AddVector(inventory, prefix + ".k_b", mha.k_b);
  AddVector(inventory, prefix + ".v_w", mha.v_w);
  AddVector(inventory, prefix + ".v_b", mha.v_b);
  AddVector(inventory, prefix + ".dense_w", mha.dense_w);
  AddVector(inventory, prefix + ".dense_b", mha.dense_b);
  if (mha.has_smolgen)
    AddSmolgen(inventory, prefix + ".smolgen", mha.smolgen);
}

void AddFFN(NetworkWeightInventory &inventory, const std::string &prefix,
            const BaseWeights::FFN &ffn) {
  AddVector(inventory, prefix + ".dense1_w", ffn.dense1_w);
  AddVector(inventory, prefix + ".dense1_b", ffn.dense1_b);
  AddVector(inventory, prefix + ".dense2_w", ffn.dense2_w);
  AddVector(inventory, prefix + ".dense2_b", ffn.dense2_b);
}

void AddEncoderLayer(NetworkWeightInventory &inventory,
                     const std::string &prefix,
                     const BaseWeights::EncoderLayer &layer) {
  AddMHA(inventory, prefix + ".mha", layer.mha);
  AddVector(inventory, prefix + ".ln1_gammas", layer.ln1_gammas);
  AddVector(inventory, prefix + ".ln1_betas", layer.ln1_betas);
  AddFFN(inventory, prefix + ".ffn", layer.ffn);
  AddVector(inventory, prefix + ".ln2_gammas", layer.ln2_gammas);
  AddVector(inventory, prefix + ".ln2_betas", layer.ln2_betas);
}

void AddBodyWeights(NetworkWeightInventory &inventory,
                    const MultiHeadWeights &weights) {
  AddConvBlock(inventory, "body.input", weights.input);
  AddVector(inventory, "body.ip_emb_preproc_w", weights.ip_emb_preproc_w);
  AddVector(inventory, "body.ip_emb_preproc_b", weights.ip_emb_preproc_b);
  AddVector(inventory, "body.ip_emb_w", weights.ip_emb_w);
  AddVector(inventory, "body.ip_emb_b", weights.ip_emb_b);
  AddVector(inventory, "body.ip_emb_ln_gammas", weights.ip_emb_ln_gammas);
  AddVector(inventory, "body.ip_emb_ln_betas", weights.ip_emb_ln_betas);
  AddVector(inventory, "body.ip_mult_gate", weights.ip_mult_gate);
  AddVector(inventory, "body.ip_add_gate", weights.ip_add_gate);
  AddFFN(inventory, "body.ip_emb_ffn", weights.ip_emb_ffn);
  AddVector(inventory, "body.ip_emb_ffn_ln_gammas",
            weights.ip_emb_ffn_ln_gammas);
  AddVector(inventory, "body.ip_emb_ffn_ln_betas",
            weights.ip_emb_ffn_ln_betas);
  AddVector(inventory, "body.smolgen_w", weights.smolgen_w);

  for (std::size_t i = 0; i < weights.encoder.size(); ++i) {
    AddEncoderLayer(inventory, "body.encoder." + std::to_string(i),
                    weights.encoder[i]);
  }
  for (std::size_t i = 0; i < weights.residual.size(); ++i) {
    const auto &res = weights.residual[i];
    const std::string prefix = "body.residual." + std::to_string(i);
    AddConvBlock(inventory, prefix + ".conv1", res.conv1);
    AddConvBlock(inventory, prefix + ".conv2", res.conv2);
    if (res.has_se) {
      AddVector(inventory, prefix + ".se.w1", res.se.w1);
      AddVector(inventory, prefix + ".se.b1", res.se.b1);
      AddVector(inventory, prefix + ".se.w2", res.se.w2);
      AddVector(inventory, prefix + ".se.b2", res.se.b2);
    }
  }
}

void AddMovesLeftWeights(NetworkWeightInventory &inventory,
                         const MultiHeadWeights &weights) {
  AddConvBlock(inventory, "moves_left", weights.moves_left);
  AddVector(inventory, "moves_left.ip_mov_w", weights.ip_mov_w);
  AddVector(inventory, "moves_left.ip_mov_b", weights.ip_mov_b);
  AddVector(inventory, "moves_left.ip1_mov_w", weights.ip1_mov_w);
  AddVector(inventory, "moves_left.ip1_mov_b", weights.ip1_mov_b);
  AddVector(inventory, "moves_left.ip2_mov_w", weights.ip2_mov_w);
  AddVector(inventory, "moves_left.ip2_mov_b", weights.ip2_mov_b);
}

void AddPolicyHeadWeights(NetworkWeightInventory &inventory,
                          const MultiHeadWeights::PolicyHead &head,
                          const std::string &head_name) {
  const std::string prefix = "policy." + head_name;
  AddConvBlock(inventory, prefix + ".policy1", head.policy1);
  AddConvBlock(inventory, prefix + ".policy", head.policy);
  AddVector(inventory, prefix + ".ip_pol_w", head.ip_pol_w);
  AddVector(inventory, prefix + ".ip_pol_b", head.ip_pol_b);
  AddVector(inventory, prefix + ".ip2_pol_w", head.ip2_pol_w);
  AddVector(inventory, prefix + ".ip2_pol_b", head.ip2_pol_b);
  AddVector(inventory, prefix + ".ip3_pol_w", head.ip3_pol_w);
  AddVector(inventory, prefix + ".ip3_pol_b", head.ip3_pol_b);
  AddVector(inventory, prefix + ".ip4_pol_w", head.ip4_pol_w);
  for (std::size_t i = 0; i < head.pol_encoder.size(); ++i) {
    AddEncoderLayer(inventory, prefix + ".encoder." + std::to_string(i),
                    head.pol_encoder[i]);
  }
}

void AddValueHeadWeights(NetworkWeightInventory &inventory,
                         const MultiHeadWeights::ValueHead &head,
                         const std::string &head_name) {
  const std::string prefix = "value." + head_name;
  AddConvBlock(inventory, prefix + ".value", head.value);
  AddVector(inventory, prefix + ".ip_val_w", head.ip_val_w);
  AddVector(inventory, prefix + ".ip_val_b", head.ip_val_b);
  AddVector(inventory, prefix + ".ip1_val_w", head.ip1_val_w);
  AddVector(inventory, prefix + ".ip1_val_b", head.ip1_val_b);
  AddVector(inventory, prefix + ".ip2_val_w", head.ip2_val_w);
  AddVector(inventory, prefix + ".ip2_val_b", head.ip2_val_b);
  AddVector(inventory, prefix + ".ip_val_err_w", head.ip_val_err_w);
  AddVector(inventory, prefix + ".ip_val_err_b", head.ip_val_err_b);
}

} // namespace

std::size_t NetworkWeightInventory::TotalElements() const {
  std::size_t total = 0;
  for (const auto &tensor : tensors)
    total += tensor.elements;
  return total;
}

std::size_t NetworkWeightInventory::TotalBytes() const {
  return TotalElements() * sizeof(float);
}

bool NetworkWeightInventory::Contains(std::string_view name) const {
  for (const auto &tensor : tensors) {
    if (tensor.name == name)
      return true;
  }
  return false;
}

std::string NetworkWeightInventory::Summary() const {
  std::ostringstream out;
  out << tensors.size() << " tensors, " << TotalElements()
      << " float elements, " << TotalBytes() << " bytes";
  return out.str();
}

NetworkWeightInventory CreateNetworkWeightInventory(
    const MultiHeadWeights &weights, const std::string &policy_head,
    const std::string &value_head, const NetworkTensorPlan &plan) {
  NetworkWeightInventory inventory;
  AddBodyWeights(inventory, weights);

  const auto policy_it = weights.policy_heads.find(policy_head);
  if (policy_it == weights.policy_heads.end()) {
    throw std::runtime_error("selected policy head is missing: " +
                             policy_head);
  }
  AddPolicyHeadWeights(inventory, policy_it->second, policy_head);

  const auto value_it = weights.value_heads.find(value_head);
  if (value_it == weights.value_heads.end()) {
    throw std::runtime_error("selected value head is missing: " + value_head);
  }
  AddValueHeadWeights(inventory, value_it->second, value_head);

  if (plan.moves_left)
    AddMovesLeftWeights(inventory, weights);

  return inventory;
}

} // namespace NN
} // namespace MetalFish
