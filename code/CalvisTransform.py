import torch
import numpy as np


class TwoDToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        # force a 3d
        # DO NOT FORGET to check on this if the image alredy have 3D!
        image = image[..., np.newaxis]
        # print(image)
        # numpy image: H x W x C
        # torch image: C X H X W
        inputs = image.transpose((2, 0, 1))
        # print(inputs.shape)

        dimensions = np.array(
            [
                sample["annotations"]["human_dimensions"][human_dim]
                for human_dim in sample["annotations"]["human_dimensions"]
            ]
        )

        # Use this parameter to control the tensor dtype.
        # Swap coordinate axis because
        tensor_dtype = torch.float32
        return {
            "image": torch.tensor(inputs, dtype=tensor_dtype),
            "annotations": {
                "human_dimensions": torch.tensor(
                    [dimensions], dtype=tensor_dtype
                )
            },
            "imagefile": sample["imagefile"],
            "annotation_file": sample["annotation_file"],
        }


if __name__ == "__main__":
    simulated_sample = {
        "image": np.random.randint(low=0, high=254, size=(5, 7), dtype=np.int),
        "annotations": {"human_dimensions": {"dim1": 1, "dim2": 2, "dim3": 3}},
        "imagefile": "simulated_image_filename.png",
        "annotation_file": "simulated_annotation_filename.json",
    }
    transform = TwoDToTensor()

    print(type(simulated_sample))
    print(type(simulated_sample["image"]))
    print(type(simulated_sample["annotations"]["human_dimensions"]))
    print(simulated_sample["annotations"]["human_dimensions"])

    print(simulated_sample["image"].shape)

    trasformed_sample = transform(simulated_sample)

    print(type(simulated_sample))
    print(type(trasformed_sample["image"]))
    print(type(trasformed_sample["annotations"]["human_dimensions"]))
    print(trasformed_sample["annotations"]["human_dimensions"])

    print(trasformed_sample["image"].shape)
    print(trasformed_sample["annotations"]["human_dimensions"].shape)
