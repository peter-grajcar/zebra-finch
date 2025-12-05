#!/usr/bin/env python3
import nemo.collections.asr as nemo_asr
import torch.onnx
from nemo.core import typecheck

typecheck.set_typecheck_enabled(False)

model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

# B - batch size
# C - feature dimension (MEL spectrogram)
# D - encoder output dimension
# T - time dimension
# L - number of RNN layers
# H - RNN hidden dimension
# U - label sequence length

# [B, C, T]
processed_signal = torch.randn((1, 128, 1234))

# [B]
processed_signal_len = torch.randn((1))

# [B, U, H]
decoder_labels = torch.randn((1, 1))

# ([L, B, H], [L, B, H])
decoder_state = torch.randn((2, 1, 640)), torch.randn((2, 1, 640))

# [B, T, D]
# transposed
encoder_output = torch.randn((1, 1234, 1024))

# [B, U, H]
decoder_output = torch.randn((1, 1, 640))

# [B, 1, H]
projected_encoder_output_slice = torch.randn((1, 1, 640))

# [B, L, H]
projected_decoder_output = torch.randn((1, 1, 640))

# PyTorch sucks so much
model.decoder.forward = model.decoder.predict
torch.onnx.export(
    model.decoder,
    (
        decoder_labels,
        decoder_state,
        False,
    ),
    output_names=("g", "state_0", "state_1"),
    external_data=True,
    f="weights/parakeet/decoder.predict.onnx",
    dynamo=True,
)

model.joint.forward = model.joint.project_encoder
torch.onnx.export(
    model.joint,
    (encoder_output,),
    external_data=True,
    f="weights/parakeet/joint.project_encoder.onnx",
    dynamo=True,
)

model.joint.forward = model.joint.project_prednet
torch.onnx.export(
    model.joint,
    (decoder_output,),
    external_data=True,
    f="weights/parakeet/joint.project_prednet.onnx",
    dynamo=True,
)

model.joint.forward = model.joint.joint_after_projection
torch.onnx.export(
    model.joint,
    (
        projected_encoder_output_slice,
        projected_decoder_output,
    ),
    external_data=True,
    f="weights/parakeet/joint.joint_after_projection.onnx",
    dynamo=True,
)

torch.onnx.export(
    model.encoder,
    (
        processed_signal,
        processed_signal_len,
    ),
    external_data=True,
    f="weights/parakeet/encoder.forward.onnx",
    dynamo=True,
)
