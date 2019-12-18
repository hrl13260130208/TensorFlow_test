import tensorflow as tf

tf.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker hosts")
tf.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # create cluster
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # create the server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    server.join()

if __name__ == "__main__":
    tf.app.run()