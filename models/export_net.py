import PIDNet.models as models
from PIDNet.configs import config, update_config
import argparse
import torch
import os
from openvino.inference_engine import IECore


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=r"C:\Users\eviatarsegev\Desktop\Projects\SkyDetector\PIDNet\configs\skyground\pidnet_large_SkyGround.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    print("start")
    args = parse_args()

    # Load pre-trained PyTorch model
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)
    # Set the model to evaluation mode
    model.eval()
    # Provide sample input dimensions (adjust based on your model's input size and shape)
    dummy_input = torch.randn(1, 3, 480, 640)
    # Provide a filename for the exported ONNX model
    onnx_filename = "C:/Users/eviatarsegev/Desktop/Projects/SkyDetector/PIDNet/models/pidnet_l_skyground.onnx"
    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_filename, export_params=True)

    # Check if the file exists
    if os.path.exists(onnx_filename):
        print("ONNX model successfully exported.")
    else:
        print("Exporting the ONNX model failed.")

    # Initialize OpenVINO Inference Engine
    ie = IECore()

    # Load the ONNX model using the OpenVINO API
    net = ie.read_network(model=onnx_filename)

    # Convert the ONNX model to an OpenVINO IR model
    exec_net = ie.load_network(network=net, device_name="CPU")

    # Save the converted OpenVINO IR model
    ir_model_filename = "C:/Users/eviatarsegev/Desktop/Projects/SkyDetector/PIDNet/models/pidnet_l_skyground.xml"
    exec_net.export(ir_model_filename)

    print("OpenVINO IR model exported successfully.")


if __name__ == '__main__':
    main()
