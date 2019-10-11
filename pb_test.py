import tensorflow as tf
from jrsinput import *

IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_DEPTH = 3
NUM_CLASS = 4

def test(pb_path, test_image, test_labels):
    '''
    This function is used to evaluate the test data. Please finish pre-precessing in advance

    :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
    img_depth]
    :return: the softmax probability with shape [num_test_images, num_labels]
    '''
    num_test_images = len(test_image)
    num_batches = num_test_images // FLAGS.test_batch_size
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
            output_tensor_name = sess.graph.get_tensor_by_name("Softmax:0")
            pre_out = np.array([]).reshape(-1, NUM_CLASS)
            labels = np.array([], 'int')
            for step in range(num_batches):
                if step % 5 == 0:
                    print('%i batches finished!' % step)
                # offset = np.random.choice(124 - FLAGS.test_batch_size, 1)[0]
                offset = step * FLAGS.test_batch_size
                test_image_batch = test_image[offset:offset + FLAGS.test_batch_size, ...]
                test_image_batch = random_crop_and_flip(test_image_batch, padding_size=FLAGS.padding_size)
                test_image_batch = whitening_image(test_image_batch)
                test_label = test_labels[offset:offset + FLAGS.test_batch_size, ...]
                out = sess.run(output_tensor_name, feed_dict={input_image_tensor: test_image_batch})
                pre_out = np.concatenate((pre_out, out))
                labels = np.concatenate((labels, test_label))
            class_id = tf.argmax(pre_out, 1)
            correct_predition = tf.equal(class_id, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_predition, 'float'))
            print(sess.run(accuracy))
def main():
    word2number_dict = {"0": 0, "1": 1, "2": 2, "3": 3}
    test_image, test_labels = read_validation_data(word2number_dict, FLAGS.padding_size)
    pb_path = 'F:\\ResNet-JRSdatabase\\ResNet_V2_DG\\accuracy=0.98\\frozen_model.pb'
    print('Loading the test data...')
    print('----------------------------')
    test(pb_path, test_image, test_labels)
if __name__=='__main__':
    main()