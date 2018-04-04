    #Implement simple 1-layer network tensorflow
    #Will check slim.conv2d

    import tensorrt as trt
    import pycuda.driver as cuda
    import tensorflow as tf
    import numpy as np
    from tensorflow.python.framework import graph_util
    import uff
    from tensorrt.parsers import uffparser

    slim = tf.contrib.slim

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    def sumArray(array):
        holder=0
        for i in range(len(array)):
            holder=holder+array[i]
        return holder


    def isclose(a, b, rel_tol=1e-05, abs_tol=0.00003):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def compare_arrays(array1,array2):
        if(len(array1)!=len(array2)):
            return False
        for i in range(len(array1)):
            status=isclose(array1[i],array2[i])
            if(status==False):
                return False
        return True


    def init_weights(shape):
        """ Weight initialization """
        weights = np.ones(shape)

        #native
        #weights=tf.ones(shape)
        return weights

    def get_data():
        inputs=np.random.rand(1,28,28,3)
        return inputs

    def forward_prop(inputs):
        #native
        ones=init_weights([3,3,3,32])
        net=tf.nn.conv2d(inputs,ones,strides=[1,2,2,1],padding='SAME')
        return net


        #tf-slim
        # ones=init_weights([3,3,32])
        # weights_init=tf.constant_initializer(ones)
        #
        # with slim.arg_scope([slim.conv2d], padding='VALID',weights_initializer=weights_init):
        #     net=slim.conv2d(inputs, 32,[3, 3],stride=2)
        #     return net

    def main():
        train_X=get_data()

        tensorrt_input=train_X.reshape(3,28,28)

        tensorrt_input=tensorrt_input.astype(np.float32)
        X = tf.placeholder("float", shape=[1, 28, 28, 3])
        h_conv1=forward_prop(X)

        # saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        tf.train.write_graph(sess.graph_def, '.', 'hellotensor.pbtxt')

        final_result=sess.run(h_conv1,feed_dict={X:train_X})

        # print(final_result)

        #saver.save(sess, './hellotensor.ckpt')

        output_graph_name='./hellotensor.pb'
        output_node_names='Conv2D'

        output_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names.split(","))
        output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)

        uff_model = uff.from_tensorflow(output_graph_def, output_nodes=['Conv2D'])
        dump = open('slimConv.uff', 'wb')
        dump.write(uff_model)
        dump.close()

        # with tf.gfile.GFile(output_graph_name, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())

        uff_model = open("/home/dami/TensorRt_test/slimConv.uff", 'rb').read()
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        parser = uffparser.create_uff_parser()
        parser.register_input("Placeholder", (3, 28, 28), 0)
        parser.register_output("Conv2D")

        engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)

        parser.destroy()

        runtime = trt.infer.create_infer_runtime(G_LOGGER)
        context = engine.create_execution_context()

        dims_data = engine.get_binding_dimensions(0).to_DimsCHW()
        dims_out1 = engine.get_binding_dimensions(1).to_DimsCHW()

        _out0 = np.empty(dims_data.C() * dims_data.H() * dims_data.W(), dtype=np.float32)
        _out1 = np.empty(dims_out1.C() * dims_out1.H() * dims_out1.W(), dtype=np.float32)

        d_out0 = cuda.mem_alloc(1 * dims_data.C() * dims_data.H() * dims_data.W() * _out0.dtype.itemsize)
        d_out1 = cuda.mem_alloc(1 * dims_out1.C() * dims_out1.H() * dims_out1.W() * _out1.dtype.itemsize)

        bindings = [int(d_out0), int(d_out1)]

        stream = cuda.Stream()

        # transfer input data to device
        cuda.memcpy_htod_async(d_out0, tensorrt_input, stream)
        # execute model
        context.enqueue(1, bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(_out1, d_out1, stream)
        # synchronize threads
        stream.synchronize()

        # re_array=_out1.reshape((13, 13, 32))

        if (_out1.shape != final_result.shape):
            results = final_result.reshape(_out1.shape)

        print(str(compare_arrays(results, _out1)))
        print(sumArray(_out1))
        print(sumArray(results))

        context.destroy()
        engine.destroy()
        runtime.destroy()



    if __name__ == '__main__':
        main()