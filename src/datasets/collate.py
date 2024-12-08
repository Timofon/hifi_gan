import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    dataset_padded = dict()

    for key in dataset_items[0].keys():
        if key == "path":
            dataset_padded[key] = [item[key] for item in dataset_items]
        elif key == "spectrogram":
            # freq_length = dataset_items[0]["spectrogram"].shape[1]
            # max_time_length = max(
            #     item["spectrogram"].shape[2] for item in dataset_items
            # )
            # dataset_padded["spectrogram"] = torch.zeros(
            #     (len(dataset_items), freq_length, max_time_length)
            # )
            # for i, item in enumerate(dataset_items):
            #     current_length = item["spectrogram"].shape[2]
            #     dataset_padded["spectrogram"][i, :, :current_length] = item[
            #         "spectrogram"
            #     ][0]
            dataset_padded[key] = torch.stack([item[key].squeeze(0) for item in dataset_items])
        elif key == "audio":
            # max_audio_length = max(len(item["audio"][0]) for item in dataset_items)

            # dataset_padded[key] = torch.zeros((len(dataset_items), max_audio_length))
            # for i, item in enumerate(dataset_items):
            #     current_length = len(item["audio"][0])
            #     dataset_padded[key][i, :current_length] = item["audio"][0]
            dataset_padded[key] = torch.stack([item[key] for item in dataset_items])

    return dataset_padded
