#!/usr/bin/env python3
import onnx

model = onnx.load("weights/parakeet/decoder.predict.onnx")


for input in model.graph.input:
    print(input.name)
    for dim in input.type.tensor_type.shape.dim:
        if dim.dim_param:
            dim.dim_param = dim.dim_param.replace("s10", "batch_size")
            print("  -", dim.dim_param)

for value_info in model.graph.value_info:
    print(value_info.name)
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dim.dim_param = dim.dim_param.replace("s10", "batch_size")
            print("  -", dim.dim_param)

# for node in model.graph.node:
#     print(node.name)

for output in model.graph.output:
    print(input.name)
    for dim in output.type.tensor_type.shape.dim:
        if dim.dim_param:
            dim.dim_param = dim.dim_param.replace("s10", "batch_size")
            print("  -", dim.dim_param)

onnx.save(model, "weights/parakeet/decoder_renamed.predict.onnx")
