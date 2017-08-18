import tensorflow as tf
import numpy as np
import aggregate_tensors as at
"""
Class which handles logic of aggregating tensors
both synchronously and asynchronously
"""

class Communicator:

    def __init__(self):
        self._lib = at._comm_lib
        self.num_workers = tf.slice(self.get_info(), [0], [1])
        self.rank = tf.slice(self.get_info(),[1], [1])
        self.local_rank = tf.slice(self.get_info(),[2], [1])

        self._num_elements = None

    # initializes mpi
    def start(self):

        return self._lib.aggregate_tensors_init()
    
    # returns tensors of length 2 [world_size, rank]
    def get_info(self):
        return self._lib.aggregate_tensors_get_info()

    # finalize MPI
    def end(self):
        return self._lib.aggregate_tensors_finalize()


    # given a list of tensor returns all tensors
    def aggregate_tensors(self,tensors):
        num_elements = self.get_num_elements(tensors)

        out_tensors = self._lib.aggregate_tensors(tensors
                , num_elements)

        return out_tensors

    def get_num_elements(self, tensors):
        shapes = [tensor.get_shape().as_list() for tensor in tensors]
        return sum([np.product(shape) for shape in shapes])

    def gather_tensors(self, tensors, num_proc):
        num_elements = [self.get_num_elements([tensor]) for
                tensor in tensors]
        out_tensors = []
        prev_tensor = self._lib.gather_tensors(tensors[0] , num_proc,
                num_elements[0])
        out_tensors.append(prev_tensor)

        for num_element, tensor in zip(num_elements[1:],
                tensors[1:]):

            with tf.control_dependencies(prev_tensor):
                prev_tensor =  self._lib.gather_tensors(tensor,num_proc,
                    num_element)

            out_tensors.append(prev_tensor)
        return zip(*out_tensors)

