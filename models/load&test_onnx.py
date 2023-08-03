# import cv2
#
# # Load the ONNX model using OpenCV
# onnx_model_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\PIDNet\models\pidnet_l_skyground.onnx"
# net = cv2.dnn.readNet(onnx_model_path)
#
# # Load and preprocess the input image (replace 'image_path' with your image file path)
# image_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\mid-air-dataset\Kite_training\sunny\color_left\trajectory_0004\000000.JPEG"
# image = cv2.imread(image_path)
#
# # Resize the image to the input size required by the model
# input_size = (640 , 480)  # Adjust based on your model's input size
# input_blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=input_size, mean=(0, 0, 0), swapRB=True, crop=False)
#
# # Set the input to the network
# net.setInput(input_blob)
#
# # Perform inference and get the output from the network
# output = net.forward()
#
# cv2.imwrite("output.jpg",output)
# # Process the output (Replace this with your desired post-processing logic)
# # Output processing depends on the specific model you are using.
# # The 'output' variable will contain the results of the inference.
# # Process the 'output' to obtain the final detections or predictions.

# import cv2
# from openvino.inference_engine import IECore
#
# # Load the ONNX model using OpenCV
# onnx_model_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\PIDNet\models\pidnet_l_skyground.onnx"
# net = cv2.dnn.readNet(onnx_model_path)
#
# # Initialize the OpenVINO Inference Engine
# ie = IECore()
#
# # Read the ONNX model using the Inference Engine
# net = ie.read_network(model=onnx_model_path, weights=None)
#
# # Get the input layer name and shape
# input_layer_name = next(iter(net.input_info))
# input_layer_shape = net.input_info[input_layer_name].input_data.shape
#
# # Set the input size for preprocessing (replace this with the actual input size)
# input_size = (480, 640)
#
# # Load and preprocess the input image (replace 'image_path' with your image file path)
# image_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\mid-air-dataset\Kite_training\sunny\color_left\trajectory_0004\000000.JPEG"
# image = cv2.imread(image_path)
#
# # Resize the image to the input size required by the model
# image = cv2.resize(image, input_size)
#
# # Prepare the input blob for the model
# input_blob = image.transpose((2, 0, 1))  # Change HWC to CHW format
# input_blob = input_blob.reshape((1, *input_blob.shape))
#
# # Load the model to the Inference Engine
# exec_net = ie.load_network(network=net, device_name="CPU")
#
# # Set the input for the model
# exec_net.inputs[input_layer_name] = input_blob
#
# # Perform inference and get the output
# output = exec_net.infer()
#
# # Process the output (Replace this with your desired post-processing logic)
# # Output processing depends on the specific model you are using.
# # The 'output' variable will contain the results of the inference.
# # Process the 'output' to obtain the final detections or predictions.

# import cv2
# from openvino.inference_engine import IECore
#
# # Load the ONNX model using OpenCV
# onnx_model_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\PIDNet\models\pidnet_l_skyground.onnx"
# net = cv2.dnn.readNet(onnx_model_path)
#
# # Initialize the OpenVINO Inference Engine
# ie = IECore()
#
# # Read the ONNX model using the Inference Engine
# net = ie.read_network(model=onnx_model_path, weights=None)
#
# # Get the input layer name and shape
# input_layer_name = next(iter(net.input_info))
# input_layer_shape = net.input_info[input_layer_name].input_data.shape
#
# # Set the input size for preprocessing (replace this with the actual input size)
# input_size = (224, 224)
#
# # Load and preprocess the input image (replace 'image_path' with your image file path)
# image_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\mid-air-dataset\Kite_training\sunny\color_left\trajectory_0004\000000.JPEG"
# image = cv2.imread(image_path)
#
# # Resize the image to the input size required by the model
# image = cv2.resize(image, input_size)
#
# # Prepare the input blob for the model
# input_blob = image.transpose((2, 0, 1))  # Change HWC to CHW format
# input_blob = input_blob.reshape((1, *input_blob.shape))
#
# # Load the model to the Inference Engine
# exec_net = ie.load_network(network=net, device_name="CPU")
#
# # Get the input and output blobs for the model
# input_blob_name = next(iter(exec_net.inputs))
# output_blob_name = next(iter(exec_net.outputs))
#
# # Set the input for the model
# exec_net.input_info[input_blob_name].precision = "U8"
# exec_net.set_blob(input_blob_name, input_blob)
#
# # Perform inference and get the output
# output = exec_net.infer(inputs={input_blob_name: input_blob})
#
# # Process the output (Replace this with your desired post-processing logic)
# # Output processing depends on the specific model you are using.
# # The 'output' variable will contain the results of the inference.
# # Process the 'output' to obtain the final detections or predictions.

import cv2
from openvino.inference_engine import IECore

# Load the ONNX model using OpenCV
onnx_model_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\PIDNet\models\pidnet_l_skyground.onnx"
net = cv2.dnn.readNet(onnx_model_path)

# Initialize the OpenVINO Inference Engine
ie = IECore()

# Read the ONNX model using the Inference Engine
net = ie.read_network(model=onnx_model_path, weights=None)

# Get the input layer name and shape
input_layer_name = next(iter(net.input_info))
input_layer_shape = net.input_info[input_layer_name].input_data.shape

# Set the input size for preprocessing (replace this with the actual input size)
input_size = (640, 480)

# Load and preprocess the input image (replace 'image_path' with your image file path)
image_path = r"C:\Users\eviatarsegev\Desktop\Projects\Sky-Ground-Segmentation\mid-air-dataset\Kite_training\sunny\color_left\trajectory_0004\000000.JPEG"
image = cv2.imread(image_path)

# Resize the image to the input size required by the model
image = cv2.resize(image, input_size)

# Prepare the input blob for the model
input_blob = image.transpose((2, 0, 1))  # Change HWC to CHW format
input_blob = input_blob.reshape((1, *input_blob.shape))

# Load the model to the Inference Engine
exec_net = ie.load_network(network=net, device_name="CPU")
# Get the output layer name(s)
output_layer_name = next(iter(net.outputs))
# Set the input for the model using the infer() method
output = exec_net.infer(inputs={input_layer_name: input_blob})
output_data = output[output_layer_name]
print("finished")
# Process the output (Replace this with your desired post-processing logic)
# Output processing depends on the specific model you are using.
# The 'output' variable will contain the results of the inference.
# Process the 'output' to obtain the final detections or predictions.


