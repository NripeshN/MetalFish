/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  Licensed under GPL-3.0
  
  Note: This file uses manual memory management (ARC disabled).
  Metal objects are explicitly retained/released.
*/

#include "metal_network.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Metal {

namespace {

// Helper to convert activation function enum to string
std::string ActivationToString(MetalFishNN::NetworkFormat::ActivationFunction act) {
  switch (act) {
    case MetalFishNN::NetworkFormat::ACTIVATION_RELU:
      return "relu";
    case MetalFishNN::NetworkFormat::ACTIVATION_MISH:
      return "mish";
    case MetalFishNN::NetworkFormat::ACTIVATION_SWISH:
      return "swish";
    case MetalFishNN::NetworkFormat::ACTIVATION_RELU_2:
      return "relu_2";
    case MetalFishNN::NetworkFormat::ACTIVATION_SELU:
      return "selu";
    case MetalFishNN::NetworkFormat::ACTIVATION_TANH:
      return "tanh";
    case MetalFishNN::NetworkFormat::ACTIVATION_SIGMOID:
      return "sigmoid";
    default:
      return "relu";
  }
}

// Apply activation function using MPSGraph
MPSGraphTensor* ApplyActivation(MPSGraph* graph, MPSGraphTensor* input,
                               NSString* activation, NSString* name) {
  if ([activation isEqualToString:@"relu"]) {
    return [graph reLUWithTensor:input name:name];
  } else if ([activation isEqualToString:@"relu_2"]) {
    // ReLU squared
    auto relu = [graph reLUWithTensor:input name:[name stringByAppendingString:@"/relu"]];
    return [graph squareWithTensor:relu name:name];
  } else if ([activation isEqualToString:@"swish"]) {
    // Swish: x * sigmoid(x)
    auto sigmoid = [graph sigmoidWithTensor:input name:[name stringByAppendingString:@"/sigmoid"]];
    return [graph multiplicationWithPrimaryTensor:input
                                   secondaryTensor:sigmoid
                                              name:name];
  } else if ([activation isEqualToString:@"mish"]) {
    // Mish: x * tanh(softplus(x))
    auto softplus = [graph softPlusWithTensor:input name:[name stringByAppendingString:@"/softplus"]];
    auto tanh = [graph tanhWithTensor:softplus name:[name stringByAppendingString:@"/tanh"]];
    return [graph multiplicationWithPrimaryTensor:input
                                   secondaryTensor:tanh
                                              name:name];
  } else if ([activation isEqualToString:@"tanh"]) {
    return [graph tanhWithTensor:input name:name];
  } else if ([activation isEqualToString:@"sigmoid"]) {
    return [graph sigmoidWithTensor:input name:name];
  } else if ([activation isEqualToString:@"selu"]) {
    // SELU: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
    auto zero = [graph constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    auto pos = [graph maximumWithPrimaryTensor:input secondaryTensor:zero name:[name stringByAppendingString:@"/pos"]];
    auto exp = [graph exponentWithTensor:input name:[name stringByAppendingString:@"/exp"]];
    auto exp_minus_1 = [graph subtractionWithPrimaryTensor:exp
                                           secondaryTensor:[graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32]
                                                      name:[name stringByAppendingString:@"/exp_m1"]];
    auto alpha_exp = [graph multiplicationWithPrimaryTensor:exp_minus_1
                                            secondaryTensor:[graph constantWithScalar:1.67326 dataType:MPSDataTypeFloat32]
                                                       name:[name stringByAppendingString:@"/alpha_exp"]];
    auto neg = [graph minimumWithPrimaryTensor:input secondaryTensor:zero name:[name stringByAppendingString:@"/neg"]];
    auto cond_neg = [graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:input secondaryTensor:zero name:nil]
                                    truePredicateTensor:alpha_exp
                                   falsePredicateTensor:zero
                                                   name:[name stringByAppendingString:@"/cond"]];
    auto sum = [graph additionWithPrimaryTensor:pos secondaryTensor:cond_neg name:[name stringByAppendingString:@"/sum"]];
    return [graph multiplicationWithPrimaryTensor:sum
                                  secondaryTensor:[graph constantWithScalar:1.0507 dataType:MPSDataTypeFloat32]
                                             name:name];
  }
  return input;  // No activation
}

}  // anonymous namespace

// Implementation class
class MetalNetwork::Impl {
public:
  Impl(const WeightsFile& weights);
  ~Impl();
  
  NetworkOutput Evaluate(const InputPlanes& input);
  std::vector<NetworkOutput> EvaluateBatch(const std::vector<InputPlanes>& inputs);
  std::string GetNetworkInfo() const;

private:
  void BuildGraph(const WeightsFile& weights);
  MPSGraphTensor* BuildEmbedding(const WeightsFile& weights);
  MPSGraphTensor* BuildEncoderStack(MPSGraphTensor* input, const WeightsFile& weights);
  MPSGraphTensor* BuildEncoderLayer(MPSGraphTensor* input, 
                                    const MetalFishNN::Weights::EncoderLayer& layer,
                                    int layer_idx);
  MPSGraphTensor* BuildMultiHeadAttention(MPSGraphTensor* input,
                                         const MetalFishNN::Weights::MHA& mha,
                                         int layer_idx);
  MPSGraphTensor* BuildFFN(MPSGraphTensor* input,
                          const MetalFishNN::Weights::FFN& ffn,
                          int layer_idx);
  MPSGraphTensor* BuildLayerNorm(MPSGraphTensor* input,
                                const MetalFishNN::Weights::Layer& gammas,
                                const MetalFishNN::Weights::Layer& betas,
                                NSString* name);
  MPSGraphTensor* BuildPolicyHead(MPSGraphTensor* input, const WeightsFile& weights);
  MPSGraphTensor* BuildValueHead(MPSGraphTensor* input, const WeightsFile& weights);
  
  MPSGraphTensor* CreateConstant(const MetalFishNN::Weights::Layer& layer, 
                                NSArray<NSNumber*>* shape);
  
  id<MTLDevice> device_;
  id<MTLCommandQueue> commandQueue_;
  MPSGraph* graph_;
  MPSGraphTensor* inputPlaceholder_;
  MPSGraphTensor* policyOutput_;
  MPSGraphTensor* valueOutput_;
  MPSGraphTensor* wdlOutput_;
  
  int embeddingSize_;
  int numLayers_;
  int numHeads_;
  int ffnSize_;
  bool hasWDL_;
  bool hasMovesLeft_;
  
  std::string defaultActivation_;
  std::string ffnActivation_;
  std::string smolgenActivation_;
};

MetalNetwork::Impl::Impl(const WeightsFile& weights) {
  @autoreleasepool {
    // Get default Metal device
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      throw std::runtime_error("Metal is not supported on this device");
    }
    
    // Create command queue
    commandQueue_ = [device_ newCommandQueue];
    if (!commandQueue_) {
      throw std::runtime_error("Failed to create Metal command queue");
    }
    
    // Create graph
    graph_ = [[MPSGraph alloc] init];
    
    // Extract network parameters
    const auto& format = weights.format().network_format();
    
    // Determine activation functions
    if (format.has_default_activation()) {
      defaultActivation_ = (format.default_activation() == 
          MetalFishNN::NetworkFormat::DEFAULT_ACTIVATION_MISH) ? "mish" : "relu";
    } else {
      defaultActivation_ = "relu";
    }
    
    if (format.has_ffn_activation()) {
      ffnActivation_ = ActivationToString(format.ffn_activation());
    } else {
      ffnActivation_ = defaultActivation_;
    }
    
    if (format.has_smolgen_activation()) {
      smolgenActivation_ = ActivationToString(format.smolgen_activation());
    } else {
      smolgenActivation_ = "swish";
    }
    
    // Check for WDL and moves left
    hasWDL_ = (format.output() == MetalFishNN::NetworkFormat::OUTPUT_WDL ||
               format.value() == MetalFishNN::NetworkFormat::VALUE_WDL);
    hasMovesLeft_ = (format.moves_left() == MetalFishNN::NetworkFormat::MOVES_LEFT_V1);
    
    // Extract embedding size from weights
    const auto& w = weights.weights();
    if (w.has_ip_emb_b()) {
      embeddingSize_ = w.ip_emb_b().params().size() / 4;  // Assuming FLOAT32
    } else {
      embeddingSize_ = 256;  // Default
    }
    
    numLayers_ = w.encoder_size();
    numHeads_ = w.has_headcount() ? w.headcount() : 8;
    ffnSize_ = embeddingSize_ * 4;  // Typical transformer FFN size
    
    // Build the graph
    BuildGraph(weights);
  }
}

MetalNetwork::Impl::~Impl() {
  @autoreleasepool {
    // Release Metal objects (manual memory management, ARC disabled)
    if (graph_) [graph_ release];
    if (commandQueue_) [commandQueue_ release];
    if (device_) [device_ release];
  }
}

MPSGraphTensor* MetalNetwork::Impl::CreateConstant(
    const MetalFishNN::Weights::Layer& layer,
    NSArray<NSNumber*>* shape) {
  
  if (!layer.has_params() || layer.params().empty()) {
    throw std::runtime_error("Layer has no parameters");
  }
  
  // Decode layer to float vector
  FloatVector data = DecodeLayer(layer);
  
  // Create NSData from float vector
  NSData* nsdata = [NSData dataWithBytes:data.data() 
                                  length:data.size() * sizeof(float)];
  
  // Create MPSGraphTensorData
  MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc]
      initWithDevice:device_
                data:nsdata
               shape:shape
            dataType:MPSDataTypeFloat32];
  
  // Create constant tensor
  MPSGraphTensor* tensor = [graph_ constantWithData:tensorData
                                              shape:shape
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
  
  [tensorData release];
  return tensor;
}

MPSGraphTensor* MetalNetwork::Impl::BuildLayerNorm(
    MPSGraphTensor* input,
    const MetalFishNN::Weights::Layer& gammas,
    const MetalFishNN::Weights::Layer& betas,
    NSString* name) {
  
  // Layer normalization: (x - mean) / sqrt(variance + epsilon) * gamma + beta
  NSArray<NSNumber*>* axes = @[@-1];  // Normalize over last dimension
  
  auto mean = [graph_ meanOfTensor:input axes:axes name:[name stringByAppendingString:@"/mean"]];
  auto variance = [graph_ varianceOfTensor:input axes:axes name:[name stringByAppendingString:@"/var"]];
  
  // Add epsilon for numerical stability
  auto epsilon = [graph_ constantWithScalar:1e-5 dataType:MPSDataTypeFloat32];
  auto var_eps = [graph_ additionWithPrimaryTensor:variance
                                   secondaryTensor:epsilon
                                              name:[name stringByAppendingString:@"/var_eps"]];
  
  // Standard deviation
  auto stddev = [graph_ squareRootWithTensor:var_eps
                                        name:[name stringByAppendingString:@"/stddev"]];
  
  // Normalize
  auto centered = [graph_ subtractionWithPrimaryTensor:input
                                       secondaryTensor:mean
                                                  name:[name stringByAppendingString:@"/centered"]];
  auto normalized = [graph_ divisionWithPrimaryTensor:centered
                                      secondaryTensor:stddev
                                                 name:[name stringByAppendingString:@"/normalized"]];
  
  // Scale and shift
  auto gammaSize = gammas.params().size() / 4;  // FLOAT32
  auto gammasTensor = CreateConstant(gammas, @[@(gammaSize)]);
  auto betasTensor = CreateConstant(betas, @[@(gammaSize)]);
  
  auto scaled = [graph_ multiplicationWithPrimaryTensor:normalized
                                        secondaryTensor:gammasTensor
                                                   name:[name stringByAppendingString:@"/scaled"]];
  auto shifted = [graph_ additionWithPrimaryTensor:scaled
                                   secondaryTensor:betasTensor
                                              name:name];
  
  return shifted;
}

MPSGraphTensor* MetalNetwork::Impl::BuildMultiHeadAttention(
    MPSGraphTensor* input,
    const MetalFishNN::Weights::MHA& mha,
    int layer_idx) {
  
  NSString* name = [NSString stringWithFormat:@"encoder_%d/mha", layer_idx];
  
  // Q, K, V projections
  auto qWeights = CreateConstant(mha.q_w(), @[@(embeddingSize_), @(embeddingSize_)]);
  auto qBias = CreateConstant(mha.q_b(), @[@(embeddingSize_)]);
  auto kWeights = CreateConstant(mha.k_w(), @[@(embeddingSize_), @(embeddingSize_)]);
  auto kBias = CreateConstant(mha.k_b(), @[@(embeddingSize_)]);
  auto vWeights = CreateConstant(mha.v_w(), @[@(embeddingSize_), @(embeddingSize_)]);
  auto vBias = CreateConstant(mha.v_b(), @[@(embeddingSize_)]);
  
  // Project to Q, K, V
  auto Q = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                          secondaryTensor:qWeights
                                                     name:[name stringByAppendingString:@"/q_proj"]];
  Q = [graph_ additionWithPrimaryTensor:Q secondaryTensor:qBias
                                   name:[name stringByAppendingString:@"/q"]];
  
  auto K = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                          secondaryTensor:kWeights
                                                     name:[name stringByAppendingString:@"/k_proj"]];
  K = [graph_ additionWithPrimaryTensor:K secondaryTensor:kBias
                                   name:[name stringByAppendingString:@"/k"]];
  
  auto V = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                          secondaryTensor:vWeights
                                                     name:[name stringByAppendingString:@"/v_proj"]];
  V = [graph_ additionWithPrimaryTensor:V secondaryTensor:vBias
                                   name:[name stringByAppendingString:@"/v"]];
  
  // Reshape for multi-head: [batch, seq, embed] -> [batch, seq, heads, head_dim]
  int headDim = embeddingSize_ / numHeads_;
  
  // For simplicity, implement single-head attention (can be extended to multi-head)
  // Scaled dot-product attention: softmax(Q*K^T / sqrt(d)) * V
  auto KT = [graph_ transposeTensor:K dimension:-1 withDimension:-2
                               name:[name stringByAppendingString:@"/k_t"]];
  
  auto scores = [graph_ matrixMultiplicationWithPrimaryTensor:Q
                                              secondaryTensor:KT
                                                         name:[name stringByAppendingString:@"/scores"]];
  
  // Scale by sqrt(head_dim)
  float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  auto scaleTensor = [graph_ constantWithScalar:scale dataType:MPSDataTypeFloat32];
  scores = [graph_ multiplicationWithPrimaryTensor:scores
                                   secondaryTensor:scaleTensor
                                              name:[name stringByAppendingString:@"/scaled_scores"]];
  
  // Softmax
  auto attn = [graph_ softMaxWithTensor:scores axis:-1
                                   name:[name stringByAppendingString:@"/attn"]];
  
  // Apply attention to V
  auto output = [graph_ matrixMultiplicationWithPrimaryTensor:attn
                                              secondaryTensor:V
                                                         name:[name stringByAppendingString:@"/attn_out"]];
  
  // Output projection
  auto outWeights = CreateConstant(mha.dense_w(), @[@(embeddingSize_), @(embeddingSize_)]);
  auto outBias = CreateConstant(mha.dense_b(), @[@(embeddingSize_)]);
  
  output = [graph_ matrixMultiplicationWithPrimaryTensor:output
                                         secondaryTensor:outWeights
                                                    name:[name stringByAppendingString:@"/out_proj"]];
  output = [graph_ additionWithPrimaryTensor:output
                             secondaryTensor:outBias
                                        name:name];
  
  return output;
}

MPSGraphTensor* MetalNetwork::Impl::BuildFFN(
    MPSGraphTensor* input,
    const MetalFishNN::Weights::FFN& ffn,
    int layer_idx) {
  
  NSString* name = [NSString stringWithFormat:@"encoder_%d/ffn", layer_idx];
  
  // First linear layer
  int ffnHiddenSize = ffn.dense1_b().params().size() / 4;  // FLOAT32
  auto w1 = CreateConstant(ffn.dense1_w(), @[@(embeddingSize_), @(ffnHiddenSize)]);
  auto b1 = CreateConstant(ffn.dense1_b(), @[@(ffnHiddenSize)]);
  
  auto hidden = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                              secondaryTensor:w1
                                                         name:[name stringByAppendingString:@"/fc1"]];
  hidden = [graph_ additionWithPrimaryTensor:hidden
                             secondaryTensor:b1
                                        name:[name stringByAppendingString:@"/fc1_bias"]];
  
  // Activation
  NSString* actName = [NSString stringWithUTF8String:ffnActivation_.c_str()];
  hidden = ApplyActivation(graph_, hidden, actName,
                          [name stringByAppendingString:@"/activation"]);
  
  // Second linear layer
  auto w2 = CreateConstant(ffn.dense2_w(), @[@(ffnHiddenSize), @(embeddingSize_)]);
  auto b2 = CreateConstant(ffn.dense2_b(), @[@(embeddingSize_)]);
  
  auto output = [graph_ matrixMultiplicationWithPrimaryTensor:hidden
                                              secondaryTensor:w2
                                                         name:[name stringByAppendingString:@"/fc2"]];
  output = [graph_ additionWithPrimaryTensor:output
                             secondaryTensor:b2
                                        name:name];
  
  return output;
}

MPSGraphTensor* MetalNetwork::Impl::BuildEncoderLayer(
    MPSGraphTensor* input,
    const MetalFishNN::Weights::EncoderLayer& layer,
    int layer_idx) {
  
  // Pre-norm architecture: LayerNorm -> MHA -> Residual
  auto ln1 = BuildLayerNorm(input, layer.ln1_gammas(), layer.ln1_betas(),
                           [NSString stringWithFormat:@"encoder_%d/ln1", layer_idx]);
  
  auto mha = BuildMultiHeadAttention(ln1, layer.mha(), layer_idx);
  
  // Residual connection
  auto residual1 = [graph_ additionWithPrimaryTensor:input
                                     secondaryTensor:mha
                                                name:[NSString stringWithFormat:@"encoder_%d/res1", layer_idx]];
  
  // Pre-norm architecture: LayerNorm -> FFN -> Residual
  auto ln2 = BuildLayerNorm(residual1, layer.ln2_gammas(), layer.ln2_betas(),
                           [NSString stringWithFormat:@"encoder_%d/ln2", layer_idx]);
  
  auto ffn = BuildFFN(ln2, layer.ffn(), layer_idx);
  
  // Residual connection
  auto residual2 = [graph_ additionWithPrimaryTensor:residual1
                                     secondaryTensor:ffn
                                                name:[NSString stringWithFormat:@"encoder_%d/res2", layer_idx]];
  
  return residual2;
}

MPSGraphTensor* MetalNetwork::Impl::BuildEncoderStack(
    MPSGraphTensor* input,
    const WeightsFile& weights) {
  
  const auto& w = weights.weights();
  MPSGraphTensor* x = input;
  
  for (int i = 0; i < numLayers_; ++i) {
    x = BuildEncoderLayer(x, w.encoder(i), i);
  }
  
  return x;
}

MPSGraphTensor* MetalNetwork::Impl::BuildEmbedding(const WeightsFile& weights) {
  const auto& w = weights.weights();
  
  // Input: [batch, 112, 64] (112 planes, 64 squares)
  // Flatten to [batch, 7168]
  auto flattened = [graph_ reshapeTensor:inputPlaceholder_
                               withShape:@[@-1, @7168]
                                    name:@"input/flatten"];
  
  // Embedding projection
  auto embWeights = CreateConstant(w.ip_emb_w(), @[@7168, @(embeddingSize_)]);
  auto embBias = CreateConstant(w.ip_emb_b(), @[@(embeddingSize_)]);
  
  auto embedded = [graph_ matrixMultiplicationWithPrimaryTensor:flattened
                                                secondaryTensor:embWeights
                                                           name:@"input/embedding"];
  embedded = [graph_ additionWithPrimaryTensor:embedded
                               secondaryTensor:embBias
                                          name:@"input/embedding_bias"];
  
  // Apply activation if specified
  NSString* actName = [NSString stringWithUTF8String:defaultActivation_.c_str()];
  embedded = ApplyActivation(graph_, embedded, actName, @"input/embedding_act");
  
  // Layer norm if present
  if (w.has_ip_emb_ln_gammas() && w.has_ip_emb_ln_betas()) {
    embedded = BuildLayerNorm(embedded, w.ip_emb_ln_gammas(), w.ip_emb_ln_betas(),
                             @"input/embedding_ln");
  }
  
  return embedded;
}

MPSGraphTensor* MetalNetwork::Impl::BuildPolicyHead(
    MPSGraphTensor* input,
    const WeightsFile& weights) {
  
  const auto& w = weights.weights();
  
  // Simple policy head: Linear projection to 1858 outputs
  if (w.has_ip_pol_w() && w.has_ip_pol_b()) {
    int policySize = w.ip_pol_b().params().size() / 4;  // Should be 1858
    
    auto weights_tensor = CreateConstant(w.ip_pol_w(), @[@(embeddingSize_), @(policySize)]);
    auto bias_tensor = CreateConstant(w.ip_pol_b(), @[@(policySize)]);
    
    auto policy = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                                secondaryTensor:weights_tensor
                                                           name:@"policy/fc"];
    policy = [graph_ additionWithPrimaryTensor:policy
                               secondaryTensor:bias_tensor
                                          name:@"policy/output"];
    
    return policy;
  }
  
  // Fallback: create dummy output
  return [graph_ constantWithScalar:0.0 shape:@[@-1, @(kPolicyOutputs)]
                           dataType:MPSDataTypeFloat32];
}

MPSGraphTensor* MetalNetwork::Impl::BuildValueHead(
    MPSGraphTensor* input,
    const WeightsFile& weights) {
  
  const auto& w = weights.weights();
  
  if (hasWDL_) {
    // WDL head: output 3 values (win, draw, loss)
    if (w.has_ip_val_w() && w.has_ip_val_b()) {
      int valueSize = w.ip_val_b().params().size() / 4;
      
      auto weights_tensor = CreateConstant(w.ip_val_w(), @[@(embeddingSize_), @(valueSize)]);
      auto bias_tensor = CreateConstant(w.ip_val_b(), @[@(valueSize)]);
      
      auto value = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                                 secondaryTensor:weights_tensor
                                                            name:@"value/fc"];
      value = [graph_ additionWithPrimaryTensor:value
                                secondaryTensor:bias_tensor
                                           name:@"value/output"];
      
      return value;
    }
  } else {
    // Single value head
    if (w.has_ip1_val_w() && w.has_ip1_val_b()) {
      auto weights_tensor = CreateConstant(w.ip1_val_w(), @[@(embeddingSize_), @1]);
      auto bias_tensor = CreateConstant(w.ip1_val_b(), @[@1]);
      
      auto value = [graph_ matrixMultiplicationWithPrimaryTensor:input
                                                 secondaryTensor:weights_tensor
                                                            name:@"value/fc"];
      value = [graph_ additionWithPrimaryTensor:value
                                secondaryTensor:bias_tensor
                                           name:@"value/output"];
      
      // Apply tanh activation for value in [-1, 1]
      value = [graph_ tanhWithTensor:value name:@"value/tanh"];
      
      return value;
    }
  }
  
  // Fallback: create dummy output
  int outputSize = hasWDL_ ? 3 : 1;
  return [graph_ constantWithScalar:0.0 shape:@[@-1, @(outputSize)]
                           dataType:MPSDataTypeFloat32];
}

void MetalNetwork::Impl::BuildGraph(const WeightsFile& weights) {
  @autoreleasepool {
    // Create input placeholder: [batch, 112, 64]
    inputPlaceholder_ = [graph_ placeholderWithShape:@[@-1, @(kTotalPlanes), @64]
                                            dataType:MPSDataTypeFloat32
                                                name:@"input"];
    
    // Build embedding
    auto embedded = BuildEmbedding(weights);
    
    // Build encoder stack (transformer layers)
    auto encoded = BuildEncoderStack(embedded, weights);
    
    // Build policy head
    policyOutput_ = BuildPolicyHead(encoded, weights);
    
    // Build value head
    valueOutput_ = BuildValueHead(encoded, weights);
    
    if (hasWDL_) {
      wdlOutput_ = valueOutput_;  // Same as value for WDL networks
    }
  }
}

NetworkOutput MetalNetwork::Impl::Evaluate(const InputPlanes& input) {
  return EvaluateBatch({input})[0];
}

std::vector<NetworkOutput> MetalNetwork::Impl::EvaluateBatch(
    const std::vector<InputPlanes>& inputs) {
  
  @autoreleasepool {
    int batchSize = static_cast<int>(inputs.size());
    
    // Prepare input data: [batch, 112, 64]
    std::vector<float> inputData(batchSize * kTotalPlanes * 64);
    for (int b = 0; b < batchSize; ++b) {
      for (int p = 0; p < kTotalPlanes; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
          inputData[b * kTotalPlanes * 64 + p * 64 + sq] = inputs[b][p][sq];
        }
      }
    }
    
    // Create input tensor data
    NSData* inputNSData = [NSData dataWithBytes:inputData.data()
                                         length:inputData.size() * sizeof(float)];
    MPSGraphTensorData* inputTensorData = [[MPSGraphTensorData alloc]
        initWithDevice:device_
                  data:inputNSData
                 shape:@[@(batchSize), @(kTotalPlanes), @64]
              dataType:MPSDataTypeFloat32];
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
    
    // Run inference
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder_: inputTensorData
    };
    
    NSArray<MPSGraphTensor*>* targetTensors = @[policyOutput_, valueOutput_];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = 
        [graph_ runWithMTLCommandQueue:commandQueue_
                                 feeds:feeds
                        targetTensors:targetTensors
                     targetOperations:nil];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Extract policy output
    MPSGraphTensorData* policyData = results[policyOutput_];
    NSData* policyNSData = [policyData mpsndarray].data;
    const float* policyPtr = static_cast<const float*>([policyNSData bytes]);
    
    // Extract value output
    MPSGraphTensorData* valueData = results[valueOutput_];
    NSData* valueNSData = [valueData mpsndarray].data;
    const float* valuePtr = static_cast<const float*>([valueNSData bytes]);
    
    // Convert to NetworkOutput
    std::vector<NetworkOutput> outputs;
    outputs.reserve(batchSize);
    
    for (int b = 0; b < batchSize; ++b) {
      NetworkOutput output;
      
      // Copy policy
      output.policy.resize(kPolicyOutputs);
      std::memcpy(output.policy.data(), policyPtr + b * kPolicyOutputs,
                  kPolicyOutputs * sizeof(float));
      
      // Copy value
      if (hasWDL_) {
        output.has_wdl = true;
        output.wdl[0] = valuePtr[b * 3 + 0];  // Win
        output.wdl[1] = valuePtr[b * 3 + 1];  // Draw
        output.wdl[2] = valuePtr[b * 3 + 2];  // Loss
        output.value = output.wdl[0] - output.wdl[2];  // Q = W - L
      } else {
        output.has_wdl = false;
        output.value = valuePtr[b];
      }
      
      outputs.push_back(output);
    }
    
    [inputTensorData release];
    
    return outputs;
  }
}

std::string MetalNetwork::Impl::GetNetworkInfo() const {
  std::ostringstream oss;
  oss << "Metal Neural Network\n";
  oss << "  Device: " << [[device_ name] UTF8String] << "\n";
  oss << "  Embedding size: " << embeddingSize_ << "\n";
  oss << "  Transformer layers: " << numLayers_ << "\n";
  oss << "  Attention heads: " << numHeads_ << "\n";
  oss << "  FFN size: " << ffnSize_ << "\n";
  oss << "  WDL: " << (hasWDL_ ? "Yes" : "No") << "\n";
  oss << "  Moves left: " << (hasMovesLeft_ ? "Yes" : "No") << "\n";
  oss << "  Default activation: " << defaultActivation_ << "\n";
  oss << "  FFN activation: " << ffnActivation_;
  return oss.str();
}

// MetalNetwork public interface
MetalNetwork::MetalNetwork(const WeightsFile& weights)
    : impl_(std::make_unique<Impl>(weights)) {}

MetalNetwork::~MetalNetwork() = default;

NetworkOutput MetalNetwork::Evaluate(const InputPlanes& input) {
  return impl_->Evaluate(input);
}

std::vector<NetworkOutput> MetalNetwork::EvaluateBatch(
    const std::vector<InputPlanes>& inputs) {
  return impl_->EvaluateBatch(inputs);
}

std::string MetalNetwork::GetNetworkInfo() const {
  return impl_->GetNetworkInfo();
}

}  // namespace Metal
}  // namespace NN
}  // namespace MetalFish
