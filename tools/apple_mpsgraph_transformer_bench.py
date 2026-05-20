#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


SOURCE_TEMPLATE = r"""
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "src/nn/metal/mps/NetworkGraph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static std::vector<float> RandomVector(size_t size, float stddev,
                                       uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, stddev);
  std::vector<float> values(size);
  for (float &value : values)
    value = dist(rng);
  return values;
}

static std::vector<float> FillVector(size_t size, float value) {
  return std::vector<float>(size, value);
}

static double Percentile(std::vector<double> values, double pct) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const size_t idx = std::min(values.size() - 1,
                              static_cast<size_t>((values.size() - 1) * pct));
  return values[idx];
}

int main() {
  @autoreleasepool {
    constexpr NSUInteger batch = __BATCH__;
    constexpr NSUInteger tokens = 64;
    constexpr bool toyNetwork = __TOY_NETWORK__;
    constexpr NSUInteger inputChannels = __INPUT_CHANNELS__;
    constexpr NSUInteger channels = __CHANNELS__;
    constexpr NSUInteger heads = __HEADS__;
    constexpr NSUInteger ffnChannels = __FFN_CHANNELS__;
    constexpr NSUInteger layers = __LAYERS__;
    constexpr NSUInteger policyChannels = __POLICY_CHANNELS__;
    constexpr NSUInteger valueOutputs = __VALUE_OUTPUTS__;
    constexpr int warmup = __WARMUP__;
    constexpr int iterations = __ITERATIONS__;
    const NSUInteger inputFeatureChannels = toyNetwork ? inputChannels : channels;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::cerr << "No Metal device available\n";
      return 2;
    }

    MetalNetworkGraph *graph = [[MetalNetworkGraph alloc] initWithDevice:device];
    MPSGraphTensor *input =
        [graph placeholderWithShape:@[ @(batch), @(tokens), @(inputFeatureChannels) ]
                            dataType:MPSDataTypeFloat32
                                name:@"input"];

    auto embedW = RandomVector(channels * inputChannels, 0.02f, 10);
    auto embedB = FillVector(channels, 0.0f);
    auto ln1Gamma = FillVector(channels, 1.0f);
    auto ln1Beta = FillVector(channels, 0.0f);
    auto ln2Gamma = FillVector(channels, 1.0f);
    auto ln2Beta = FillVector(channels, 0.0f);
    auto qW = RandomVector(channels * channels, 0.02f, 11);
    auto qB = FillVector(channels, 0.0f);
    auto kW = RandomVector(channels * channels, 0.02f, 12);
    auto kB = FillVector(channels, 0.0f);
    auto vW = RandomVector(channels * channels, 0.02f, 13);
    auto vB = FillVector(channels, 0.0f);
    auto oW = RandomVector(channels * channels, 0.02f, 14);
    auto oB = FillVector(channels, 0.0f);
    auto ffn1W = RandomVector(ffnChannels * channels, 0.02f, 15);
    auto ffn1B = FillVector(ffnChannels, 0.0f);
    auto ffn2W = RandomVector(channels * ffnChannels, 0.02f, 16);
    auto ffn2B = FillVector(channels, 0.0f);
    auto policyW = RandomVector(policyChannels * channels, 0.02f, 17);
    auto policyB = FillVector(policyChannels, 0.0f);
    auto valueW = RandomVector(valueOutputs * channels, 0.02f, 18);
    auto valueB = FillVector(valueOutputs, 0.0f);

    MPSGraphTensor *layer = input;
    if (toyNetwork) {
      layer = [graph addFullyConnectedLayerWithParent:layer
                                       outputChannels:channels
                                              weights:embedW.data()
                                               biases:embedB.data()
                                           activation:nil
                                                label:@"embedding"];
    }
    for (NSUInteger layerIdx = 0; layerIdx < layers; ++layerIdx) {
      NSString *prefix =
          [NSString stringWithFormat:@"layer_%lu", (unsigned long)layerIdx];
      MPSGraphTensor *norm =
          [graph addLayerNormalizationWithParent:layer
                           scaledSecondaryTensor:nil
                                          gammas:ln1Gamma.data()
                                           betas:ln1Beta.data()
                                           alpha:1.0f
                                         epsilon:1e-5f
                                           label:[prefix stringByAppendingString:@"/ln1"]];
      MPSGraphTensor *q =
          [graph addFullyConnectedLayerWithParent:norm
                                   outputChannels:channels
                                          weights:qW.data()
                                           biases:qB.data()
                                       activation:nil
                                            label:[prefix stringByAppendingString:@"/q"]];
      MPSGraphTensor *k =
          [graph addFullyConnectedLayerWithParent:norm
                                   outputChannels:channels
                                          weights:kW.data()
                                           biases:kB.data()
                                       activation:nil
                                            label:[prefix stringByAppendingString:@"/k"]];
      MPSGraphTensor *v =
          [graph addFullyConnectedLayerWithParent:norm
                                   outputChannels:channels
                                          weights:vW.data()
                                           biases:vB.data()
                                       activation:nil
                                            label:[prefix stringByAppendingString:@"/v"]];
      MPSGraphTensor *attn =
          [graph scaledMHAMatmulWithQueries:q
                                   withKeys:k
                                 withValues:v
                                      heads:heads
                                     parent:norm
                                    smolgen:nil
                          smolgenActivation:nil
                                      label:[prefix stringByAppendingString:@"/mha"]];
      MPSGraphTensor *projected =
          [graph addFullyConnectedLayerWithParent:attn
                                   outputChannels:channels
                                          weights:oW.data()
                                           biases:oB.data()
                                       activation:nil
                                            label:[prefix stringByAppendingString:@"/out"]];
      MPSGraphTensor *residual =
          [graph additionWithPrimaryTensor:layer
                           secondaryTensor:projected
                                      name:[prefix stringByAppendingString:@"/residual"]];
      MPSGraphTensor *ffn =
          [graph addLayerNormalizationWithParent:residual
                           scaledSecondaryTensor:nil
                                          gammas:ln2Gamma.data()
                                           betas:ln2Beta.data()
                                           alpha:1.0f
                                         epsilon:1e-5f
                                           label:[prefix stringByAppendingString:@"/ln2"]];
      ffn = [graph addFullyConnectedLayerWithParent:ffn
                                     outputChannels:ffnChannels
                                            weights:ffn1W.data()
                                             biases:ffn1B.data()
                                         activation:@"swish"
                                              label:[prefix stringByAppendingString:@"/ffn1"]];
      ffn = [graph addFullyConnectedLayerWithParent:ffn
                                     outputChannels:channels
                                            weights:ffn2W.data()
                                             biases:ffn2B.data()
                                         activation:nil
                                              label:[prefix stringByAppendingString:@"/ffn2"]];
      layer = [graph additionWithPrimaryTensor:residual
                               secondaryTensor:ffn
                                          name:[prefix stringByAppendingString:@"/output"]];
    }
    NSArray *targets;
    if (toyNetwork) {
      MPSGraphTensor *policy =
          [graph addFullyConnectedLayerWithParent:layer
                                   outputChannels:policyChannels
                                          weights:policyW.data()
                                           biases:policyB.data()
                                       activation:nil
                                            label:@"policy"];
      MPSGraphTensor *pooled =
          [graph meanOfTensor:layer axes:@[ @1 ] name:@"value/pool"];
      MPSGraphTensor *value =
          [graph addFullyConnectedLayerWithParent:pooled
                                   outputChannels:valueOutputs
                                          weights:valueW.data()
                                           biases:valueB.data()
                                       activation:nil
                                            label:@"value"];
      targets = @[ policy, value ];
    } else {
      targets = @[ layer ];
    }

    MPSGraphDevice *graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    auto inputValues =
        RandomVector(batch * tokens * inputFeatureChannels, 1.0f, 21);
    NSData *inputData =
        [NSData dataWithBytesNoCopy:inputValues.data()
                              length:inputValues.size() * sizeof(float)
                        freeWhenDone:NO];
    MPSGraphTensorData *inputTensorData =
        [[MPSGraphTensorData alloc] initWithDevice:graphDevice
                                              data:inputData
                                             shape:@[ @(batch), @(tokens), @(inputFeatureChannels) ]
                                          dataType:MPSDataTypeFloat32];
    NSDictionary *feeds = @{input : inputTensorData};

    auto runOnce = [&]() {
      MPSCommandBuffer *commandBuffer =
          [MPSCommandBuffer commandBufferFromCommandQueue:queue];
      MPSGraphExecutionDescriptor *descriptor =
          [[MPSGraphExecutionDescriptor alloc] init];
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      descriptor.completionHandler =
          ^(MPSGraphTensorDataDictionary *resultDictionary, NSError *error) {
            if (error != nil) {
              NSLog(@"MPSGraph benchmark error: %@", error);
            }
            dispatch_semaphore_signal(semaphore);
          };
      [graph encodeToCommandBuffer:commandBuffer
                              feeds:feeds
                      targetTensors:targets
                   targetOperations:nil
                executionDescriptor:descriptor];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    };

    for (int i = 0; i < warmup; ++i)
      runOnce();

    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      const auto start = std::chrono::steady_clock::now();
      runOnce();
      const auto end = std::chrono::steady_clock::now();
      latencies.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }

    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double mean = latencies.empty() ? 0.0 : sum / latencies.size();
    const double median = Percentile(latencies, 0.50);
    const double p90 = Percentile(latencies, 0.90);
    const double minLatency =
        latencies.empty() ? 0.0 : *std::min_element(latencies.begin(), latencies.end());
    const double maxLatency =
        latencies.empty() ? 0.0 : *std::max_element(latencies.begin(), latencies.end());

    std::cout << "{"
              << "\"model\":\"" << (toyNetwork ? "toy-network" : "transformer") << "\","
              << "\"batch\":" << batch << ","
              << "\"tokens\":" << tokens << ","
              << "\"input_channels\":" << inputChannels << ","
              << "\"channels\":" << channels << ","
              << "\"heads\":" << heads << ","
              << "\"ffn_channels\":" << ffnChannels << ","
              << "\"layers\":" << layers << ","
              << "\"policy_channels\":" << policyChannels << ","
              << "\"value_outputs\":" << valueOutputs << ","
              << "\"iterations\":" << iterations << ","
              << "\"median_ms\":" << median << ","
              << "\"mean_ms\":" << mean << ","
              << "\"p90_ms\":" << p90 << ","
              << "\"min_ms\":" << minLatency << ","
              << "\"max_ms\":" << maxLatency << "}" << std::endl;
  }
  return 0;
}
"""


def run_command(
    args: list[str], timeout: float = 120.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def render_source(args: argparse.Namespace) -> str:
    if args.tokens != 64:
        raise ValueError("MPSGraph helper benchmark currently requires --tokens 64")
    if args.channels % args.heads != 0:
        raise ValueError("--channels must be divisible by --heads")
    if args.layers < 1:
        raise ValueError("--layers must be at least 1")
    if args.input_channels < 1:
        raise ValueError("--input-channels must be at least 1")
    if args.policy_channels < 1:
        raise ValueError("--policy-channels must be at least 1")
    if args.value_outputs < 1:
        raise ValueError("--value-outputs must be at least 1")
    return (
        SOURCE_TEMPLATE.replace("__BATCH__", str(args.batch))
        .replace("__TOY_NETWORK__", "true" if args.model == "toy-network" else "false")
        .replace("__INPUT_CHANNELS__", str(args.input_channels))
        .replace("__CHANNELS__", str(args.channels))
        .replace("__HEADS__", str(args.heads))
        .replace("__FFN_CHANNELS__", str(args.channels * args.ffn_mult))
        .replace("__LAYERS__", str(args.layers))
        .replace("__POLICY_CHANNELS__", str(args.policy_channels))
        .replace("__VALUE_OUTPUTS__", str(args.value_outputs))
        .replace("__WARMUP__", str(args.warmup))
        .replace("__ITERATIONS__", str(args.iterations))
    )


def compile_benchmark(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("clang++ or c++ is required")
    proto_include = ROOT / "build" / "proto"
    if not proto_include.exists():
        raise RuntimeError("build/proto is missing; configure the CMake build first")
    return run_command(
        [
            compiler,
            "-std=c++20",
            "-fobjc-arc",
            "-ObjC++",
            "-I.",
            "-Isrc",
            "-Isrc/nn",
            "-Ibuild",
            "-Ibuild/proto",
            "-I/opt/homebrew/include",
            "src/nn/metal/mps/NetworkGraph.mm",
            str(source),
            "-L/opt/homebrew/lib",
            "-lprotobuf",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "MetalPerformanceShadersGraph",
            "-framework",
            "MetalPerformanceShaders",
            "-framework",
            "QuartzCore",
            "-o",
            str(output),
        ]
    )


def print_human(result: dict[str, object]) -> None:
    print("MetalFish MPSGraph transformer microbenchmark")
    print(
        f"  Shape:  model={result.get('model', 'transformer')} "
        f"batch={result['batch']} tokens={result['tokens']} "
        f"input_channels={result['input_channels']} channels={result['channels']} "
        f"heads={result['heads']} ffn={result['ffn_channels']} "
        f"layers={result['layers']} policy_channels={result['policy_channels']} "
        f"value_outputs={result['value_outputs']}"
    )
    print(
        "  MPSGraph: "
        f"median={result['median_ms']:.4f} ms "
        f"mean={result['mean_ms']:.4f} ms "
        f"p90={result['p90_ms']:.4f} ms"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark transformer-shaped MPSGraph kernels."
    )
    parser.add_argument(
        "--model", choices=["transformer", "toy-network"], default="transformer"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--input-channels", type=int, default=112)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--policy-channels", type=int, default=32)
    parser.add_argument("--value-outputs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep-source", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_text = render_source(args)
        with tempfile.TemporaryDirectory(prefix="metalfish-mpsgraph-bench-") as tmp:
            tmpdir = Path(tmp)
            source = Path(args.keep_source) if args.keep_source else tmpdir / "bench.mm"
            binary = tmpdir / "bench"
            source.write_text(source_text, encoding="utf-8")
            compile_result = compile_benchmark(source, binary)
            if compile_result.returncode != 0:
                print(compile_result.stdout, end="")
                print(compile_result.stderr, file=sys.stderr, end="")
                return compile_result.returncode
            run_result = run_command([str(binary)])
            if run_result.returncode != 0:
                print(run_result.stdout, end="")
                print(run_result.stderr, file=sys.stderr, end="")
                return run_result.returncode
            result = json.loads(run_result.stdout)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
