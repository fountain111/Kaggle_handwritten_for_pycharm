import tensorflow as tf
import pandas as pd
import numpy as np
import Hand_written

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/minist',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def train():
    train_datas, train_labels, validation_datas, validation_labels = Hand_written.split_data()
    with tf.Graph().as_default():
        global_step = tf.Variable(0)


        model_labels = Hand_written.inference(train_datas)

        loss = Hand_written.loss(model_labels,train_labels)

        optimizer = Hand_written.train(loss)

        tf.scalar_summary("loss",loss)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
    for step in range(FLAGS.max_steps):
        _,loss_value = sess.run([optimizer,feed_dict={x: batch_xs, y_: batch_ys])

        #if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        print("step=", step, loss_value)





def main(argv=None):  # pylint: disable=unused-argument
    train()



if __name__ == '__main__':
  tf.app.run()