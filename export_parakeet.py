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
processed_signal_len = torch.zeros((1), dtype=torch.int32)

# [B, U, H]
decoder_labels = torch.randint(1, 8192, (1, 1), dtype=torch.int32)

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


batch_size = torch.export.Dim("batch_size", min=1)

# PyTorch sucks so much
model.decoder.forward = model.decoder.predict
# for some mysterious reason, the batch_size is called s10 instead of batch_size, only in this export...
torch.onnx.export(
    model.decoder,
    (
        decoder_labels,
        decoder_state,
        False,
    ),
    # output_names=("g", "state_0", "state_1"), # setting the output names will break the graph...
    external_data=True,
    f="weights/parakeet/decoder.predict.onnx",
    dynamo=True,
    dynamic_shapes={
        "y": {0: batch_size},
        "state": ({1: batch_size}, {1: batch_size}),
        "add_sos": None,
    },
)

exit()

model.joint.forward = model.joint.project_encoder
torch.onnx.export(
    model.joint,
    (encoder_output,),
    external_data=True,
    f="weights/parakeet/joint.project_encoder.onnx",
    dynamo=True,
    dynamic_shapes={
        "encoder_output": {0: batch_size, 1: torch.export.Dim("time_steps")},
    },
)

model.joint.forward = model.joint.project_prednet
torch.onnx.export(
    model.joint,
    (decoder_output,),
    external_data=True,
    f="weights/parakeet/joint.project_prednet.onnx",
    dynamo=True,
    dynamic_shapes={
        "prednet_output": {0: batch_size},
    },
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
    dynamic_shapes={
        "f": {0: batch_size},
        "g": {0: batch_size},
    },
)

# without this, the export does not work... words cannot express my hatred for pytorch
model.encoder.eval()

torch.onnx.export(
    model.encoder,
    (
        processed_signal,
        processed_signal_len,
    ),
    external_data=True,
    f="weights/parakeet/encoder.forward.onnx",
    dynamo=True,
    dynamic_shapes={
        "audio_signal": {0: batch_size, 2: torch.export.Dim('signal_length', min=9, max=8 * 625 - 6)},
        "length": {},
    },
)
