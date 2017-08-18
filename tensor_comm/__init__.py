import tensorflow as tf
import os

from .mpi_session import MPISession
from .communicator import Communicator

dir = os.path.dirname(os.path.abspath(__file__))
try:
    _comm_lib = tf.load_op_library("%s/../lib/aggregate_tensors_cpu_gpu.so" % dir)
except:
    try:
        _comm_lib = tf.load_op_library("%s/../lib/aggregate_tensors_cpu.so" % dir)
    except:
        print("Could Not Find shared object file")
        raise



