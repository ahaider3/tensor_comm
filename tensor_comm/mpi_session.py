import tensorflow as tf


class MPISession(tf.Session):


    def __init__(self, comm, graph=None, config=None):
        tf.Session.__init__(self, graph=graph, config=config)
        self._comm = comm
        self.run(self._comm.start())
        info = self.run(self._comm.get_info())
        self.num_workers = int(info[0])
        self.rank = int(info[1])

    def close(self):
        self.run(self._comm.end())
        tf.Session.close(self)



