import tensorflow as tf
from typing import List
import cv2
import os
import numpy as np

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


DATA_DIR = 'data'


def get_alignment_path(video_path):
    """Get align file path for given video path"""

    video_name = os.path.basename(video_path)  # get just video name
    align_name = video_name.replace('.mpg', '.align')  # construct align name
    return os.path.join(DATA_DIR, 'alignments', align_name)  # full path


def load_video(path: str) -> List[float]:
    # print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    # print(path)
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(video_path):

  # Load video frames
  frames = load_video(video_path)

  # Reshape here
  if len(frames) > 75:
      frames = frames[:75]

  # Get align path, load if exists
  align_path = get_alignment_path(video_path)
  if os.path.exists(align_path):
    alignments = load_alignments(align_path)
  else:
    # No alignment file found
    alignments = np.array([])

  return frames, alignments

