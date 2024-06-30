

import argparse
import tensorflow as tf

LABELS = ['#','*','0','1','2','3','4','5','6','7','8','9','A','B','C','D']

def run_prediction(path, model):
    x = path
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    result = model(waveform[tf.newaxis, :])
    prediction = result['predictions']
    max_index = tf.argmax(prediction[0])
    print(LABELS[max_index.numpy()])

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--file", "-f", help="File", type=str, default=None)
    args = args.parse_args()

    model = tf.saved_model.load("model")

    if args.file:
        run_prediction(args.file, model)
        return
if __name__ == "__main__":
    main()

