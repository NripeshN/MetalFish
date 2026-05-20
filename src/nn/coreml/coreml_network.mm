/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "coreml_network.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace MetalFish {
namespace NN {
namespace CoreML {

namespace {

NSString *ToNSString(const std::string &text) {
  return [NSString stringWithUTF8String:text.c_str()];
}

std::string FromNSString(NSString *text) {
  return text ? std::string([text UTF8String]) : std::string();
}

std::string ErrorString(NSError *error) {
  if (!error)
    return {};
  NSString *message = [error localizedDescription];
  return FromNSString(message);
}

MLComputeUnits ComputeUnitsFromName(const std::string &name) {
  if (name == "cpu")
    return MLComputeUnitsCPUOnly;
  if (name == "cpu-gpu")
    return MLComputeUnitsCPUAndGPU;
  if (name == "all")
    return MLComputeUnitsAll;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
  if (name == "cpu-ne")
    return MLComputeUnitsCPUAndNeuralEngine;
  return MLComputeUnitsCPUAndNeuralEngine;
#else
  return MLComputeUnitsAll;
#endif
}

std::string ComputeUnitsLabel(MLComputeUnits units) {
  switch (units) {
  case MLComputeUnitsCPUOnly:
    return "cpu";
  case MLComputeUnitsCPUAndGPU:
    return "cpu-gpu";
  case MLComputeUnitsAll:
    return "all";
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
  case MLComputeUnitsCPUAndNeuralEngine:
    return "cpu-ne";
#endif
  default:
    return "unknown";
  }
}

float HalfToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
  uint32_t exp = (bits >> 10) & 0x1FU;
  uint32_t mant = bits & 0x03FFU;
  uint32_t out = 0;

  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400U) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03FFU;
      out = sign | ((exp + 127U - 15U) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7F800000U | (mant << 13);
  } else {
    out = sign | ((exp + 127U - 15U) << 23) | (mant << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

float ReadMultiArrayValue(MLMultiArray *array, NSInteger offset) {
  switch (array.dataType) {
  case MLMultiArrayDataTypeFloat32:
    return static_cast<float *>(array.dataPointer)[offset];
  case MLMultiArrayDataTypeDouble:
    return static_cast<float>(static_cast<double *>(array.dataPointer)[offset]);
  case MLMultiArrayDataTypeFloat16:
    return HalfToFloat(static_cast<uint16_t *>(array.dataPointer)[offset]);
  default:
    throw std::runtime_error("Unsupported Core ML multi-array data type");
  }
}

NSInteger ShapeValue(NSArray<NSNumber *> *shape, NSUInteger idx,
                    NSInteger fallback) {
  if (!shape || idx >= [shape count])
    return fallback;
  return [shape[idx] integerValue];
}

NSInteger StrideValue(NSArray<NSNumber *> *strides, NSUInteger idx,
                     NSInteger fallback) {
  if (!strides || idx >= [strides count])
    return fallback;
  return [strides[idx] integerValue];
}

} // namespace

class CoreMLNetwork::Impl {
public:
  Impl(const WeightsFile & /*file*/, const std::string &model_path,
       const std::string &compute_units)
      : model_path_(std::filesystem::absolute(model_path).string()),
        requested_compute_units_(compute_units) {
    if (model_path.empty())
      throw std::runtime_error("Core ML backend requires NNCoreMLModelPath");
    if (!std::filesystem::exists(model_path_))
      throw std::runtime_error("Core ML model does not exist: " + model_path_);

    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = ComputeUnitsFromName(compute_units);
    actual_compute_units_ = configuration.computeUnits;

    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:ToNSString(model_path_)
                            isDirectory:YES];
    NSString *extension = [[url pathExtension] lowercaseString];
    NSURL *load_url = url;
    if (![extension isEqualToString:@"mlmodelc"]) {
      compiled_url_ = [MLModel compileModelAtURL:url error:&error];
      if (!compiled_url_)
        throw std::runtime_error("Could not compile Core ML model: " +
                                 ErrorString(error));
      load_url = compiled_url_;
    }

    model_ = [MLModel modelWithContentsOfURL:load_url
                                configuration:configuration
                                        error:&error];
    if (!model_)
      throw std::runtime_error("Could not load Core ML model: " +
                               ErrorString(error));

    NSDictionary<NSString *, MLFeatureDescription *> *inputs =
        model_.modelDescription.inputDescriptionsByName;
    if ([inputs count] != 1)
      throw std::runtime_error("Core ML model must expose exactly one input");

    input_name_ = [[inputs allKeys] firstObject];
    MLFeatureDescription *input_desc = inputs[input_name_];
    if (input_desc.type != MLFeatureTypeMultiArray)
      throw std::runtime_error("Core ML model input must be a multi-array");

    NSArray<NSNumber *> *shape = input_desc.multiArrayConstraint.shape;
    if ([shape count] != 3)
      throw std::runtime_error(
          "Core ML model input must have shape [batch, 64, 112]");
    fixed_batch_ = ShapeValue(shape, 0, 1);
    const NSInteger squares = ShapeValue(shape, 1, 0);
    const NSInteger planes = ShapeValue(shape, 2, 0);
    if (fixed_batch_ < 1 || squares != 64 || planes != kTotalPlanes)
      throw std::runtime_error(
          "Core ML model input must have shape [batch, 64, 112]");
  }

  NetworkOutput Evaluate(const InputPlanes &input) {
    std::vector<InputPlanes> inputs{input};
    return EvaluateBatch(inputs).front();
  }

  std::vector<NetworkOutput> EvaluateBatch(const std::vector<InputPlanes> &inputs) {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(inputs.size());
    for (size_t offset = 0; offset < inputs.size();
         offset += static_cast<size_t>(fixed_batch_)) {
      const size_t remaining = inputs.size() - offset;
      const size_t chunk =
          std::min(remaining, static_cast<size_t>(fixed_batch_));
      std::vector<NetworkOutput> chunk_outputs =
          EvaluateFixedChunk(inputs, offset, chunk);
      outputs.insert(outputs.end(), chunk_outputs.begin(), chunk_outputs.end());
    }
    return outputs;
  }

  std::string GetNetworkInfo() const {
    std::ostringstream oss;
    oss << "Core ML backend\n";
    oss << "Model: " << model_path_ << "\n";
    oss << "Compute units: " << ComputeUnitsLabel(actual_compute_units_);
    if (requested_compute_units_ != ComputeUnitsLabel(actual_compute_units_))
      oss << " (requested " << requested_compute_units_ << ")";
    oss << "\nBatch: " << fixed_batch_;
    return oss.str();
  }

private:
  std::vector<NetworkOutput>
  EvaluateFixedChunk(const std::vector<InputPlanes> &inputs, size_t offset,
                     size_t chunk) {
    NSError *error = nil;
    NSArray<NSNumber *> *shape = @[
      @(fixed_batch_), @(64), @(kTotalPlanes)
    ];
    MLMultiArray *input_array =
        [[MLMultiArray alloc] initWithShape:shape
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&error];
    if (!input_array)
      throw std::runtime_error("Could not allocate Core ML input: " +
                               ErrorString(error));

    FillInput(input_array, inputs, offset, chunk);

    MLFeatureValue *input_value =
        [MLFeatureValue featureValueWithMultiArray:input_array];
    NSDictionary<NSString *, MLFeatureValue *> *input_dict =
        @{input_name_ : input_value};
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                          error:&error];
    if (!provider)
      throw std::runtime_error("Could not create Core ML input provider: " +
                               ErrorString(error));

    id<MLFeatureProvider> prediction =
        [model_ predictionFromFeatures:provider error:&error];
    if (!prediction)
      throw std::runtime_error("Core ML prediction failed: " +
                               ErrorString(error));

    std::vector<NetworkOutput> outputs(chunk);
    DecodeOutputs(prediction, outputs);
    return outputs;
  }

  void FillInput(MLMultiArray *array, const std::vector<InputPlanes> &inputs,
                 size_t offset, size_t chunk) {
    float *data = static_cast<float *>(array.dataPointer);
    const NSInteger batch_stride = StrideValue(array.strides, 0, 64 * kTotalPlanes);
    const NSInteger square_stride = StrideValue(array.strides, 1, kTotalPlanes);
    const NSInteger plane_stride = StrideValue(array.strides, 2, 1);

    for (NSInteger b = 0; b < fixed_batch_; ++b) {
      const InputPlanes &planes =
          inputs[offset + std::min(static_cast<size_t>(b), chunk - 1)];
      for (int sq = 0; sq < 64; ++sq) {
        for (int plane = 0; plane < kTotalPlanes; ++plane) {
          data[b * batch_stride + sq * square_stride + plane * plane_stride] =
              planes[plane][sq];
        }
      }
    }
  }

  void DecodeOutputs(id<MLFeatureProvider> prediction,
                     std::vector<NetworkOutput> &outputs) {
    bool found_policy = false;
    bool found_wdl = false;
    bool found_scalar = false;

    for (NSString *name in prediction.featureNames) {
      MLFeatureValue *feature = [prediction featureValueForName:name];
      if (!feature || feature.type != MLFeatureTypeMultiArray)
        continue;
      MLMultiArray *array = feature.multiArrayValue;
      NSArray<NSNumber *> *shape = array.shape;
      if ([shape count] == 0)
        continue;

      const NSInteger last_dim =
          ShapeValue(shape, [shape count] - 1, static_cast<NSInteger>(array.count));
      if (last_dim == kPolicyOutputs) {
        CopyPolicy(array, outputs);
        found_policy = true;
      } else if (last_dim == 3) {
        CopyWDL(array, outputs);
        found_wdl = true;
      } else if (last_dim == 1) {
        CopyScalarValue(array, outputs);
        found_scalar = true;
      }
    }

    if (!found_policy)
      throw std::runtime_error("Core ML model did not return a policy tensor");
    if (!found_wdl && !found_scalar)
      throw std::runtime_error("Core ML model did not return a value tensor");
  }

  void CopyPolicy(MLMultiArray *array, std::vector<NetworkOutput> &outputs) {
    const NSUInteger rank = [array.shape count];
    const NSInteger batch_stride = rank >= 2 ? StrideValue(array.strides, 0, 0) : 0;
    const NSInteger feature_stride =
        StrideValue(array.strides, rank - 1, 1);
    for (size_t b = 0; b < outputs.size(); ++b) {
      for (int i = 0; i < kPolicyOutputs; ++i) {
        const NSInteger offset = static_cast<NSInteger>(b) * batch_stride +
                                 static_cast<NSInteger>(i) * feature_stride;
        outputs[b].policy[i] = ReadMultiArrayValue(array, offset);
      }
    }
  }

  void CopyWDL(MLMultiArray *array, std::vector<NetworkOutput> &outputs) {
    const NSUInteger rank = [array.shape count];
    const NSInteger batch_stride = rank >= 2 ? StrideValue(array.strides, 0, 0) : 0;
    const NSInteger feature_stride =
        StrideValue(array.strides, rank - 1, 1);
    for (size_t b = 0; b < outputs.size(); ++b) {
      NetworkOutput &out = outputs[b];
      out.has_wdl = true;
      for (int i = 0; i < 3; ++i) {
        const NSInteger offset = static_cast<NSInteger>(b) * batch_stride +
                                 static_cast<NSInteger>(i) * feature_stride;
        out.wdl[i] = ReadMultiArrayValue(array, offset);
      }
      out.value = out.wdl[0] - out.wdl[2];
    }
  }

  void CopyScalarValue(MLMultiArray *array, std::vector<NetworkOutput> &outputs) {
    const NSUInteger rank = [array.shape count];
    const NSInteger batch_stride = rank >= 2 ? StrideValue(array.strides, 0, 0) : 0;
    const NSInteger feature_stride =
        StrideValue(array.strides, rank - 1, 1);
    for (size_t b = 0; b < outputs.size(); ++b) {
      const NSInteger offset = static_cast<NSInteger>(b) * batch_stride +
                               0 * feature_stride;
      outputs[b].value = ReadMultiArrayValue(array, offset);
      outputs[b].has_wdl = false;
      outputs[b].wdl[0] = outputs[b].wdl[1] = outputs[b].wdl[2] = 0.0f;
    }
  }

  __strong MLModel *model_ = nil;
  __strong NSURL *compiled_url_ = nil;
  __strong NSString *input_name_ = nil;
  std::string model_path_;
  std::string requested_compute_units_;
  MLComputeUnits actual_compute_units_ = MLComputeUnitsCPUOnly;
  NSInteger fixed_batch_ = 1;
};

CoreMLNetwork::CoreMLNetwork(const WeightsFile &file,
                             const std::string &model_path,
                             const std::string &compute_units)
    : impl_(std::make_unique<Impl>(file, model_path, compute_units)) {}

CoreMLNetwork::~CoreMLNetwork() = default;

NetworkOutput CoreMLNetwork::Evaluate(const InputPlanes &input) {
  return impl_->Evaluate(input);
}

std::vector<NetworkOutput>
CoreMLNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  return impl_->EvaluateBatch(inputs);
}

std::string CoreMLNetwork::GetNetworkInfo() const {
  return impl_->GetNetworkInfo();
}

} // namespace CoreML
} // namespace NN
} // namespace MetalFish
