import models
from configs import config
import argparse
from configs import update_config
import torch
import os
import subprocess
import torchvision

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
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
    onnx_filename = "C:/Users/eviatarsegev/Desktop/Projects/Sky-Ground-Segmentation/PIDNet/models/pidnet_l_skyground.onnx"
    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_filename, export_params=True)

    # Check if the file exists
    if os.path.exists(onnx_filename):
        print("ONNX model successfully exported.")
    else:
        print("Exporting the ONNX model failed.")

    # Run OpenVINO Model Optimizer to generate XML and BIN files
    output_dir = "C:/Users/eviatarsegev/Desktop/Projects/Sky-Ground-Segmentation/PIDNet/models"
    model_name = "pidnet_l_skyground"
    cmd = f"mo --input_model {onnx_filename} --model_name {model_name} --output_dir {output_dir}"
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
