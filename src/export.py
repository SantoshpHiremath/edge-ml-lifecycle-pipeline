"""Export a trained model to ONNX for edge inference, and verify the
export is actually faithful -- not just "did torch.onnx.export not
raise" but a real numeric parity check between the original PyTorch
model and the exported ONNX graph run through onnxruntime, which is
the same inference engine family a real edge device would use."""

import numpy as np
import onnxruntime as ort
import torch


def export_to_onnx(model, output_path, img_size=16):
    model.eval()
    dummy_input = torch.zeros(1, 1, img_size, img_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return output_path


def verify_export_parity(model, onnx_path, sample_images, atol=1e-4):
    """Run the same input batch through both the original PyTorch model
    and the exported ONNX model, and confirm the outputs agree within a
    tight numeric tolerance. Returns (is_parity_ok, max_abs_diff)."""
    model.eval()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample_images)).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(
        ["logits"], {"image": sample_images.astype(np.float32)}
    )[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    return max_abs_diff <= atol, max_abs_diff


def onnx_predict(onnx_path, images):
    session = ort.InferenceSession(onnx_path)
    logits = session.run(["logits"], {"image": images.astype(np.float32)})[0]
    return logits.argmax(axis=1)
