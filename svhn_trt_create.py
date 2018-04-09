##############################################################################
##### Create engine from .caffemodel and .prototxt file, and save it #########
##############################################################################
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
from random import randint
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
from tensorrt import parsers

# Logging interface	for the builder, engine and runtime to handle logging message
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

#Input Output Layers
INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['loss']
#Size of image
INPUT_H = 32
INPUT_W =  32
#Number of outputs
OUTPUT_SIZE = 10

#Give the required files
MODEL_PROTOTXT = '/home/d/Desktop/model_compare_caffe/svhn/trt/svhn_trt.prototxt'
CAFFE_MODEL = '/home/d/Desktop/model_compare_caffe/svhn/trt/svhn_trt.caffemodel'
DATA = '/home/d/Desktop/model_compare_caffe/svhn/trt/trt_'
IMAGE_MEAN = '/home/d/Desktop/model_compare_caffe/svhn/trt/svhn_trt.binaryproto'

#Creating Engine by parsing caffe files
engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
                                       MODEL_PROTOTXT,
                                       CAFFE_MODEL,
                                       2000,
                                       2 << 20,
                                       OUTPUT_LAYERS,
                                       trt.infer.DataType.FLOAT)

#Getting access to image in the computer
rand_file = randint(0,10)	
path = DATA + str(rand_file) + '.png'
im = Image.open(path)
imshow(np.asarray(im))
arr = np.array(im)
img = arr.ravel()

#Pre-processing for mean image
parser = parsers.caffeparser.create_caffe_parser()
mean_blob = parser.parse_binary_proto(IMAGE_MEAN)
parser.destroy()
#NOTE: This is different than the C++ API, you must provide the size of the data
mean = mean_blob.get_data(INPUT_W ** 2)
data = np.empty([INPUT_W ** 2])
for i in range(INPUT_W ** 2):
    data[i] = float(img[i]) - mean[i]
mean_blob.destroy()

#
runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

assert(engine.get_nb_bindings() == 2)
#convert input data to Float32
img = img.astype(np.float32)
#create output array to receive data
output = np.empty(OUTPUT_SIZE, dtype = np.float32)

d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

#transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
#execute model
context.enqueue(1, bindings, stream.handle, None)
#transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
#syncronize threads
stream.synchronize()

print("Test Case: " + str(rand_file))
print ("Prediction: " + str(np.argmax(output)))
trt.utils.write_engine_to_file("new_svhn.engine", engine.serialize())
# new_engine = trt.utils.load_engine(G_LOGGER, "new_mnist.engine")

context.destroy()
engine.destroy()
# new_engine.destroy()
runtime.destroy()