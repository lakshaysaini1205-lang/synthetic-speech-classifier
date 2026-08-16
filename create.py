import os
import tensorflow as tf
import tf2onnx
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_deepfake_model.h5')
ONNX_PATH = os.path.join(BASE_DIR, 'best_deepfake_model.onnx')

print("Loading Keras model...")
model = load_model(MODEL_PATH)

print("Converting model to ONNX...")

# 1. Define input signature matching shape (batch_size, 40, 63, 1)
input_signature = [
    tf.TensorSpec(shape=(None, 40, 63, 1), dtype=tf.float32, name="input")
]

# 2. Wrap model execution in a tf.function
@tf.function(input_signature=input_signature)
def run_model(input_tensor):
  return model(input_tensor)


# 3. Convert directly using the @tf.function wrapper
model_proto, _ = tf2onnx.convert.from_function(
    run_model, input_signature=input_signature, output_path=ONNX_PATH
)

print(f"Success! Model exported to: {ONNX_PATH}")