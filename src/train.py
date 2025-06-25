from pathlib import Path
from typing import Tuple
from PIL import Image
from loguru import logger
from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from ruamel.yaml import YAML

from unet import unet_model

yaml = YAML(typ="safe")


def zoom_and_shift(
    image: np.ndarray, ground_truth: np.ndarray, max_zoom_percentage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zooms in on the image by a random amount between 0 and max_zoom_percentage,
    then shifts the image by a random amount up to the number of zoomed pixels.

    Parameters
    ----------
    image : np.ndarray
        The image to zoom and shift.
    ground_truth : np.ndarray
        The ground truth mask to zoom and shift.
    max_zoom_percentage : float
        The maximum percentage of the image size to zoom in on.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The zoomed and shifted image and ground truth mask.
    """

    # Choose a zoom percentage and caluculate the number of pixels to zoom in
    zoom = np.random.uniform(0, max_zoom_percentage)
    zoom_pixels = int(image.shape[0] * zoom)

    # If there is zoom, choose a random shift
    if int(zoom_pixels) > 0:
        shift_x = np.random.randint(int(-zoom_pixels), int(zoom_pixels))
        shift_y = np.random.randint(int(-zoom_pixels), int(zoom_pixels))

        # Zoom and shift the image
        zoomed_and_shifted_image = image[
            zoom_pixels + shift_x : -zoom_pixels + shift_x,
            zoom_pixels + shift_y : -zoom_pixels + shift_y,
        ]
        zoomed_and_shifted_ground_truth = ground_truth[
            zoom_pixels + shift_x : -zoom_pixels + shift_x,
            zoom_pixels + shift_y : -zoom_pixels + shift_y,
        ]
    else:
        # Do nothing
        shift_x = 0
        shift_y = 0

        zoomed_and_shifted_image = image
        zoomed_and_shifted_ground_truth = ground_truth

    return zoomed_and_shifted_image, zoomed_and_shifted_ground_truth


# generator for data
def image_data_generator(
    data_dir: Path,
    image_indexes: np.ndarray,
    batch_size: int,
    model_image_size: Tuple[int, int],
    image_channels: int,
    output_classes: int,
    augment_zoom_percent: float,
    augment_flip_rotate: bool,
    augment_vshift: tuple[float, float],
    norm_upper_bound: float,
    norm_lower_bound: float,
):
    """Generate batches of images and ground truth masks."""

    if image_channels != 1:
        raise NotImplementedError(
            f"Image channels {image_channels} not implemented. Only 1 channel images are supported."
        )

    while True:
        # Select files for the batch
        logger.info(f"num image indexes: {len(image_indexes)}")
        logger.info(f"batch size: {batch_size}")
        batch_indexes = np.random.choice(a=image_indexes, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Load images and ground truth
        for index in batch_indexes:
            # Load the image and ground truth
            image = np.load(data_dir / f"image_{index}.npy")
            ground_truth = np.load(data_dir / f"mask_{index}.npy").astype(bool)

            # Augment: zoom and shift
            image, ground_truth = zoom_and_shift(image, ground_truth, max_zoom_percentage=augment_zoom_percent)
            # Augment: flip & rotate
            if augment_flip_rotate:
                # 50% chance to flip
                if np.random.rand() > 0.5:
                    image = np.flip(image, axis=1)
                    ground_truth = np.flip(ground_truth, axis=1)
                # rotate to random multiple of 90 degrees
                rotation = np.random.randint(0, 4)
                image = np.rot90(image, k=rotation)
                ground_truth = np.rot90(ground_truth, k=rotation)

            # Add random vertical shift
            if augment_vshift[0] != 0 or augment_vshift[1] != 0:
                # Get random float between -augment_vshift and augment_vshift
                vshift = np.random.uniform(-augment_vshift[0], augment_vshift[1])
                # Shift the image and ground truth vertically
                if vshift > 0:
                    image += vshift

            # Resize without interpolation
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(model_image_size, resample=Image.NEAREST)
            image = np.array(pil_image)

            pil_ground_truth = Image.fromarray(ground_truth)
            pil_ground_truth = pil_ground_truth.resize(model_image_size, resample=Image.NEAREST)
            ground_truth = np.array(pil_ground_truth)

            # Normalise the image
            image = np.clip(image, norm_lower_bound, norm_upper_bound)
            image = image - norm_lower_bound
            image = image / (norm_upper_bound - norm_lower_bound)

            # Add the image and ground truth to the batch
            batch_input.append(image)
            if output_classes > 1:
                categorical_ground_truth = np.zeros(
                    shape=(model_image_size[0], model_image_size[1], output_classes)
                ).astype(np.uint8)
                # print(f"Ground truth shape: {ground_truth.shape}")
                # print(f"Categorical ground truth shape: {categorical_ground_truth.shape}")
                logger.info(
                    f"Ground truth unique values: {np.unique(ground_truth)}, counts: {np.bincount(ground_truth.flatten())}"
                )
                for i in range(output_classes):
                    # print(i)
                    categorical_ground_truth[:, :, i] = np.uint8(np.where(ground_truth == (i + 1), 1, 0))
                    logger.info(f"categorical classes and counts: {i} : {np.sum(categorical_ground_truth[:, :, i])}")
                batch_output.append(categorical_ground_truth)
            else:
                # logger.info(f"unique values in ground truth: {np.unique(ground_truth)}")
                # logger.info(f"Ground truth max value: {np.max(ground_truth)} grabbing value 2 only")
                batch_output.append(ground_truth.astype(bool))

        # Convert the lists to numpy arrays
        batch_x = np.array(batch_input).astype(np.float32)
        batch_y = np.array(batch_output).astype(np.float32)
        # np.save("batch_x.npy", batch_x)
        # np.save("batch_y.npy", batch_y)
        # raise ValueError("Debugging batch_x and batch_y")
        # logger.info(f"Batch x shape: {batch_x.shape} image channels: {image_channels}")
        # logger.info(f"Batch y shape: {batch_y.shape} output classes: {output_classes}")

        yield (batch_x, batch_y)


def train_model(
    random_seed: int,
    train_data_dir: Path,
    model_save_dir: Path,
    model_image_size: Tuple[int, int],
    image_channels: int,
    output_classes: int,
    augment_zoom_percent: float,
    augment_flip_rotate: bool,
    augment_vshift: tuple[float, float],
    conv_activation_function: str,
    final_activation_function: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    norm_upper_bound: float,
    norm_lower_bound: int,
    validation_split: float,
    loss_function: str,
):
    """Train a model to segment images."""

    logger.info("Training: Setup")

    logger.info("Training: Parameters:")
    logger.info(f"|  Random seed: {random_seed}")
    logger.info(f"|  Train data directory: {train_data_dir}")
    logger.info(f"|  Model save directory: {model_save_dir}")
    logger.info(f"|  Model image size: {model_image_size}")
    logger.info(f"|  Image channels: {image_channels}")
    logger.info(f"|  Output classes: {output_classes}")
    logger.info(f"|  Augment zoom percentage: {augment_zoom_percent}")
    logger.info(f"|  Augment flip & rotate: {augment_flip_rotate}")
    logger.info(f"|  Augment vertical shift: {augment_vshift}")
    logger.info(f"|  Conv activation function: {conv_activation_function}")
    logger.info(f"|  Final activation function: {final_activation_function}")
    logger.info(f"|  Learning rate: {learning_rate}")
    logger.info(f"|  Batch size: {batch_size}")
    logger.info(f"|  Epochs: {epochs}")
    logger.info(f"|  Normalisation upper bound: {norm_upper_bound}")
    logger.info(f"|  Normalisation lower bound: {norm_lower_bound}")
    logger.info(f"|  Validation split: {validation_split}")
    logger.info(f"|  Loss function: {loss_function}")

    # Set the random seed
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    logger.info("Training: Loading data")
    # Find the indexes of all the image files in the format of image_<index>.npy
    image_files = list(train_data_dir.glob("image_*.npy"))
    image_indexes = []
    for image_file in image_files:
        match = re.search(r"\d+", image_file.name)
        if match:
            image_indexes.append(int(match.group()))
        else:
            raise ValueError(f"Image file name {image_file.name} does not match the expected format.")
    # Find the indexes of all the mask files in the format of mask_<index>.npy
    mask_files = list(train_data_dir.glob("mask_*.npy"))
    mask_indexes = []
    for mask_file in mask_files:
        match = re.search(r"\d+", mask_file.name)
        if match:
            mask_indexes.append(int(match.group()))
        else:
            raise ValueError(f"Mask file name {mask_file.name} does not match the expected format.")

    # Check that the image and mask indexes are the same
    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    # Train test split
    train_indexes, validation_indexes = train_test_split(
        image_indexes, test_size=validation_split, random_state=random_seed
    )
    logger.info(f"Training on {len(train_indexes)} images | validating on {len(validation_indexes)} images.")

    # Create an image data generator
    logger.info("Training: Creating data generators")

    train_generator = image_data_generator(
        data_dir=train_data_dir,
        image_indexes=train_indexes,
        batch_size=batch_size,
        model_image_size=model_image_size,
        image_channels=image_channels,
        output_classes=output_classes,
        augment_zoom_percent=augment_zoom_percent,
        augment_flip_rotate=augment_flip_rotate,
        augment_vshift=augment_vshift,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
    )

    validation_generator = image_data_generator(
        data_dir=train_data_dir,
        image_indexes=validation_indexes,
        batch_size=batch_size,
        model_image_size=model_image_size,
        image_channels=image_channels,
        output_classes=output_classes,
        augment_zoom_percent=0,
        augment_flip_rotate=False,
        augment_vshift=augment_vshift,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
    )

    # Load the model
    logger.info("Training: Loading model")
    model = unet_model(
        image_height=model_image_size[0],
        image_width=model_image_size[1],
        image_channels=image_channels,
        output_classes=output_classes,
        learning_rate=learning_rate,
        conv_activation_function=conv_activation_function,
        final_activation_function=final_activation_function,
        loss_function=loss_function,
    )

    steps_per_epoch = len(train_indexes) // batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # At the end of each epoch, DVCLive will log the metrics
    logger.info("Using DVCLive to log the metrics.")
    with Live("results/train") as live:

        logger.info("Training the model.")
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch,
            verbose=1,
            callbacks=[DVCLiveCallback(live=live)],
        )

        logger.info("Training: Finished training.")

        logger.info(f"Training: Saving model to {model_save_dir}")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save(Path(model_save_dir) / "output-model.keras")
        live.log_artifact(
            str(Path(model_save_dir) / "output-model.keras"),
            type="model",
            name="output-model",
            desc="Model trained to segment cats.",
            labels=["cv", "segmentation"],
        )
        logger.info("Training: Finished.")

        # loss = history.history["loss"]
        # val_loss = history.history["val_loss"]
        # epoch_indexes = range(1, len(loss) + 1)
        # plt.plot(epoch_indexes, loss, "bo", label="Training loss")
        # plt.plot(epoch_indexes, val_loss, "b", label="Validation loss")
        # plt.title("Training and validation loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.show()

        # date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Save the model
        # model.save(model_save_dir / f"model_{date}.h5")


if __name__ == "__main__":
    logger.info("Train: Loading the parameters from the params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        train_params = all_params["train"]

    logger.info("Train: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    train_data_dir_path = Path(train_params["train_data_dir"])
    model_save_dir_path = Path(train_params["model_save_dir"])

    # Train the model
    train_model(
        random_seed=base_params["random_seed"],
        train_data_dir=train_data_dir_path,
        model_save_dir=model_save_dir_path,
        model_image_size=(base_params["model_image_size"], base_params["model_image_size"]),
        image_channels=base_params["image_channels"],
        output_classes=base_params["output_classes"],
        conv_activation_function=train_params["conv_activation_function"],
        final_activation_function=train_params["final_activation_function"],
        learning_rate=train_params["learning_rate"],
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        augment_zoom_percent=train_params["augment_zoom_percent"],
        augment_flip_rotate=train_params["augment_flip_rotate"],
        augment_vshift=train_params["augment_vshift"],
        norm_upper_bound=train_params["norm_upper_bound"],
        norm_lower_bound=train_params["norm_lower_bound"],
        validation_split=train_params["validation_split"],
        loss_function=base_params["loss_function"],
    )
