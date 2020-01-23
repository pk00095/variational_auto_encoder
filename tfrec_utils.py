import tensorflow as tf
#tf.enable_eager_execution()

import glob, os, tqdm

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape
  #mask_shape = tf.image.decode_png(mask_string).shape

  feature = {
  	  'image_raw': _bytes_feature(image_string)
      'image/height': _int64_feature(image_shape[0]),
      'image/width': _int64_feature(image_shape[1]),
      'image/depth': _int64_feature(image_shape[2]),
      'image_label': _int64_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))



def create_tfrecords(image_dir, out_path):

    assert os.path.isdir(image_dir)

    LABEL = 1

    with tf.python_io.TFRecordWriter(out_path) as writer :

      for image_file in glob.glob(os.path.join(image_dir, '*.jpg')):

            assert image_file.endswith('.jpg')
            image_string = open(image_file, 'rb').read()

            tf_example = image_example(
              image_string=image_string,
              label=LABEL)

            writer.write(tf_example.SerializeToString())


    print ('\nWritten images and mask into {}'.format(out_path))


def parse_tfrecords(filenames, height, width, shuffle=False,repeat_count=1,batch_size=32):

    def _parse_function(serialized, n_classes=n_classes):
        features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/depth': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'image_label': tf.FixedLenFeature([], tf.int64)        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
               
        image_string = parsed_example['image_raw']
        label = parsed_example['image_label']

        # decode the raw bytes so it becomes a tensor with type

        image = tf.cast(tf.image.decode_jpeg(image_string), tf.float32)
        image = tf.image.resize_images(image,(height, width))
        image.set_shape([height, width,3])

        return image, tf.cast(label,tf.float32)
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    dataset = dataset.map(_parse_function, num_parallel_calls=4)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 4)

    dataset = dataset.repeat(repeat_count) # Repeat the dataset this time
    dataset = dataset.batch(batch_size)    # Batch Size
    batch_dataset = dataset.prefetch(buffer_size=4)

    return batch_dataset

