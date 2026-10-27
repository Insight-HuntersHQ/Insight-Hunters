
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


@tf.keras.utils.register_keras_serializable(package="InsightHunters")
class ResNet50Preprocessing(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.reverse(inputs, axis=[-1]) - tf.constant(
            [103.939, 116.779, 123.68],
            dtype=tf.float32
        )

    def get_config(self):
        return super().get_config()


def load_tb_model(model_path):
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "ResNet50Preprocessing": ResNet50Preprocessing
        },
        compile=False
    )


def preprocess_image(image_input):

    if isinstance(image_input, (str, Path)):
        image_bytes = tf.io.read_file(str(image_input))

    elif isinstance(image_input, bytes):
        image_bytes = image_input

    elif hasattr(image_input, "getvalue"):
        image_bytes = image_input.getvalue()

    else:
        raise TypeError("Unsupported image input type.")

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        (224, 224),
        method="bilinear"
    )

    return tf.expand_dims(image, axis=0)


def predict_tb(model, image_input):

    image_batch = preprocess_image(image_input)

    probability = float(
        model(image_batch, training=False).numpy()[0][0]
    )

    predicted_class = (
        "TB Positive" if probability >= 0.5 else "TB Negative"
    )

    return {
        "predicted_class": predicted_class,
        "tb_probability": probability,
        "image_batch": image_batch
    }


def generate_gradcam(model, image_batch, positive_class=True):

    backbone = model.get_layer("resnet50")
    target_layer = backbone.get_layer("conv5_block3_out")

    feature_extractor = tf.keras.Model(
        backbone.input,
        [target_layer.output, backbone.output]
    )

    backbone_index = model.layers.index(backbone)
    classification_layers = model.layers[backbone_index + 1:]

    with tf.GradientTape() as tape:

        x = tf.convert_to_tensor(image_batch, dtype=tf.float32)

        for layer in model.layers[:backbone_index]:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)

        conv_outputs, x = feature_extractor(x, training=False)
        tape.watch(conv_outputs)

        for layer in classification_layers:
            x = layer(x, training=False)

        class_score = x[:, 0] if positive_class else 1.0 - x[:, 0]

    gradients = tape.gradient(class_score, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients, axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients[:, None, None, :],
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap / (
        tf.reduce_max(heatmap, axis=(1, 2), keepdims=True) + 1e-8
    )

    return heatmap.numpy()


def create_gradcam_overlay(image_batch, heatmap, alpha=0.4):

    image = image_batch[0].numpy() / 255.0

    resized_heatmap = tf.image.resize(
        heatmap[0, :, :, None],
        (224, 224),
        method="bilinear"
    ).numpy()[:, :, 0]

    coloured_heatmap = plt.get_cmap("jet")(resized_heatmap)[..., :3]

    overlay = (1 - alpha) * image + alpha * coloured_heatmap

    overlay = np.clip(overlay, 0, 1)

    return overlay


def _run_tb_inference_unchecked(model, image_input):

    prediction = predict_tb(model, image_input)

    heatmap = generate_gradcam(
        model,
        prediction["image_batch"],
        positive_class=(
            prediction["predicted_class"] == "TB Positive"
        )
    )

    overlay = create_gradcam_overlay(
        prediction["image_batch"],
        heatmap
    )

    return {
        "predicted_class": prediction["predicted_class"],
        "tb_probability": prediction["tb_probability"],
        "heatmap": heatmap,
        "overlay": overlay
    }



def load_validator_model(model_path):
    return tf.keras.models.load_model(
        str(model_path),
        compile=False
    )


def validate_chest_xray(validator_model, image_input, threshold=0.5):

    if isinstance(image_input, (str, Path)):
        image_bytes = tf.io.read_file(str(image_input))

    elif isinstance(image_input, bytes):
        image_bytes = image_input

    elif hasattr(image_input, "getvalue"):
        image_bytes = image_input.getvalue()

    else:
        raise TypeError("Unsupported image input type.")

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        (224, 224),
        method="bilinear"
    )

    image = tf.image.grayscale_to_rgb(image)

    image_batch = tf.expand_dims(image, axis=0)

    non_chest_score = float(
        validator_model(
            image_batch,
            training=False
        ).numpy()[0][0]
    )

    is_chest_xray = non_chest_score < threshold

    return {
        "is_chest_xray": is_chest_xray,
        "non_chest_score": non_chest_score,
        "threshold": threshold
    }



def run_validated_inference(
    tb_model,
    validator_model,
    image_input,
    threshold=0.5
):
    """
    Validate an uploaded image before performing TB classification.

    Returns:
        status: "accepted", "rejected", or "error"
    """

    try:
        validation = validate_chest_xray(
            validator_model,
            image_input,
            threshold=threshold
        )

    except (
        TypeError,
        ValueError,
        OSError,
        tf.errors.OpError
    ) as error:

        return {
            "status": "error",
            "message": (
                "The uploaded file could not be processed as an image. "
                "Please upload a valid chest X-ray image."
            ),
            "error_details": str(error)
        }

    if not validation["is_chest_xray"]:

        return {
            "status": "rejected",
            "message": (
                "The uploaded image does not appear to be a "
                "chest X-ray. Please upload a chest radiograph."
            ),
            "non_chest_score": validation["non_chest_score"]
        }

    prediction = _run_tb_inference_unchecked(
        tb_model,
        image_input
    )

    return {
        "status": "accepted",
        "message": "Image accepted for TB screening.",
        "non_chest_score": validation["non_chest_score"],
        "predicted_class": prediction["predicted_class"],
        "tb_probability": prediction["tb_probability"],
        "heatmap": prediction["heatmap"],
        "overlay": prediction["overlay"]
    }



def run_complete_inference(
    tb_model,
    validator_model,
    image_input,
    threshold=0.5
):
    """
    Primary inference entry point.

    Performs image validation before TB classification
    and Grad-CAM generation.
    """
    return run_validated_inference(
        tb_model,
        validator_model,
        image_input,
        threshold=threshold
    )
