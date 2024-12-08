from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms.wav_augs.spectrogram import MelSpectrogramConfig, MelSpectrogram


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, model, criterion, metrics, optimizer_discriminator, optimizer_generator, lr_scheduler_discriminator, lr_scheduler_generator, config, device, dataloaders, logger, writer, epoch_len=None, skip_oom=True, batch_transforms=None):
        super().__init__(model, criterion, metrics, optimizer_discriminator, optimizer_generator, lr_scheduler_discriminator, lr_scheduler_generator, config, device, dataloaders, logger, writer, epoch_len, skip_oom, batch_transforms)

        self.get_spectrogram = MelSpectrogram(MelSpectrogramConfig()).to(self.device)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer_discriminator.zero_grad()
            self.optimizer_generator.zero_grad()

        audio = batch["audio"]
        spectrogram = batch["spectrogram"]

        # Discriminator
        audio_predicted = self.model.generator(spectrogram)
        spectrogram_predicted = self.get_spectrogram(audio_predicted).squeeze(1)

        # Compute discriminator loss
        mpd_loss, msd_loss = self.compute_discriminators_losses(audio, audio_predicted.detach())
        discriminator_loss = mpd_loss + msd_loss
        if self.is_train:
            discriminator_loss.backward()
            self._clip_grad_norm()
            self.optimizer_discriminator.step()

        # Generator
        generator_loss = self.compute_generator_loss(audio, audio_predicted, spectrogram, spectrogram_predicted)
        if self.is_train:
            generator_loss.backward()
            self._clip_grad_norm()
            self.optimizer_generator.step()

        outputs = {
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "audio_predicted": audio_predicted
        }
        batch.update(outputs)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch
    
    def compute_discriminators_losses(self, audio, audio_predicted):
        # MPD
        ground_truths, _ = self.model.mpd(audio)
        predicted, _ = self.model.mpd(audio_predicted)
        mpd_loss = self.criterion.discriminator_loss(ground_truths, predicted)
        
        # MSD
        ground_truths, _ = self.model.msd(audio)
        predicted, _ = self.model.msd(audio_predicted)
        msd_loss = self.criterion.discriminator_loss(ground_truths, predicted)
        
        return mpd_loss, msd_loss
    
    def compute_generator_loss(self, audio, audio_predicted, spectrogram, spectrogram_predicted):
        mpd_outputs, mpd_ground_truth_features, mpd_features = self.get_scores_and_feats(self.model.mpd, audio, audio_predicted)
        msd_outputs, msd_ground_truth_features, msd_features = self.get_scores_and_feats(self.model.msd, audio, audio_predicted)

        adversarial_loss = self.criterion.adversarial_loss(mpd_outputs) + self.criterion.adversarial_loss(msd_outputs)
        mel_spec_loss = self.criterion.mel_spectrogram_loss(spectrogram, spectrogram_predicted)
        feature_matching_loss = self.criterion.feature_matching_loss(mpd_ground_truth_features, mpd_features) + self.criterion.feature_matching_loss(msd_ground_truth_features, msd_features)

        total_loss = adversarial_loss + mel_spec_loss + feature_matching_loss

        return total_loss
    
    def get_scores_and_feats(self, discriminator, audio, audio_predicted):
        _, gt_features = discriminator(audio)
        outputs, features = discriminator(audio_predicted)
        return outputs, gt_features, features

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":
            self.log_audio(audio_to_log=batch["audio"][0], name="Audio 0", **batch)
        else:
            self.log_audio(audio_to_log=batch["audio"][0], name="Train Audio 0", **batch)
            self.log_predictions(**batch)

    def log_predictions(
        self, audio_predicted, examples_to_log=1, **batch
    ):
        for i in range(examples_to_log):
            self.log_audio(audio_to_log=audio_predicted[i], name=f"Audio {i}", **batch)
    
    def log_audio(self, audio_to_log, name, **batch):
        self.writer.add_audio(name, audio_to_log[0], 22050)
