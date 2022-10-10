import os
import time
import argparse

import torch
import torch
import tensorrt as trt

torch.nn.Module.apply = lambda self, f: None
from timm.models import create_model
import onnx
from onnxsim import simplify

import models
from utils import set_seed
from deformable_attention import register_defomable_attention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str)
    parser.add_argument('-root', type=str, default='logs/inference_benchmark')
    parser.add_argument('-input_shape',
                        nargs='+',
                        type=int,
                        default=(1, 3, 224, 224))
    parser.add_argument('-seed', type=int, default=1001)
    return parser.parse_args()


def export_to_onnx(save_path, model, data):
    torch.onnx.export(
        model,
        data,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {
                0: 'batch_size'
            },
            'output': {
                0: 'batch_size'
            }
        },
        opset_version=12,
    )
    print('Start: simplify onnx')
    model = onnx.load(save_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, save_path)
    print('End: simplify onnx')


def benchmark_time(run, num_iters=1000):
    torch.cuda.synchronize()
    for _ in range(100):
        run()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iters):
        run()
    torch.cuda.synchronize()
    t2 = time.time()
    return (t2 - t1) / num_iters


def build_trt_engine(builder, network, input_shape, tf32, fp16):
    # create configuration
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        input_shape,  # min shape
        input_shape,  # optimal shape
        input_shape  # max shape
    )
    config.add_optimization_profile(profile)
    config.avg_timing_iterations = 1
    config.default_device_type = trt.DeviceType.GPU
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    # config.set_tactic_sources(trt.TacticSource.CUBLAS)
    # config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
    # config.set_tactic_sources(trt.TacticSource.CUDNN)
    # config.set_tactic_sources(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)

    # build engine
    engine = builder.build_serialized_network(network, config)
    return engine


def perf_trt_engine(engine_bytes, logger, input_data, output_gt):
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # init input and output buffer
    input_buffer = torch.zeros_like(input_data)
    output_buffer = torch.zeros_like(output_gt)
    input_binding_idx = engine.get_binding_index('input')
    output_binding_idx = engine.get_binding_index('output')
    bindings = [None, None]
    bindings[input_binding_idx] = input_buffer.data_ptr()
    bindings[output_binding_idx] = output_buffer.data_ptr()
    input_buffer.copy_(input_data)
    context.set_binding_shape(input_binding_idx, input_buffer.shape)
    torch.cuda.synchronize()

    # benchmark_time
    def _run():
        context.execute_async_v2(bindings,
                                 torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

    avg_time = benchmark_time(_run) / input_data.shape[0]
    max_diff = (output_buffer - output_gt).abs().max().item()
    return 1.0 / avg_time, max_diff


@torch.no_grad()
def main(args):
    if args.seed:
        set_seed(args.seed)
    root = os.path.join(args.root, args.model_name)
    os.makedirs(root, exist_ok=True)

    input_data = torch.randn(*args.input_shape,
                       dtype=torch.float32,
                       device=torch.device('cuda'))
    model = create_model(args.model_name, pretrained=False, num_classes=1000)
    model = model.eval().cuda()
    output_gt = model(input_data)
    torch.cuda.synchronize()

    register_defomable_attention(model)
    export_to_onnx('{}/{}.onnx'.format(root, args.model_name), model, input_data)

    # init trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)

    # create network definition
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open('{}/{}.onnx'.format(root, args.model_name), 'rb') as fr:
        parser.parse(fr.read())

    # build trt engines
    print('building tensort engine ...')
    tf32_fp16_trt_model = build_trt_engine(builder,
                                           network,
                                           args.input_shape,
                                           tf32=True,
                                           fp16=True)
    print('benchmarking tensort ...')
    throught, max_diff = perf_trt_engine(tf32_fp16_trt_model, logger,
                                         input_data, output_gt)
    print('onnx trt tf32-fp16 bz {}: \tqps: {}, \tdiff: {}'.format(
        args.input_shape[0], throught, max_diff))


if __name__ == '__main__':
    main(parse_args())