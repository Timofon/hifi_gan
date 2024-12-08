from src.metrics.base_metric import BaseMetric
from src.metrics.base_mos import Wav2Vec2MOS
import torchaudio
import torch
from torch import Tensor
from src.metrics.download_mos import download_mos
import os


class MOSMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()
        download_mos()
        self.mos_meter = Wav2Vec2MOS(path=os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt"))

    def __call__(self, audio_predicted: Tensor, **batch):
        '''audios: [B, 1, len]'''
        values = []
        for pred in audio_predicted:
            mos = self.calculate_one(pred)
            values.append(mos)
        
        return torch.mean(torch.tensor(values))
    
    def calculate_one(self, audio: Tensor):
        audio = torchaudio.functional.resample(audio.squeeze(0), 22050, 16000)
        x = self.mos_meter.processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        
        with torch.inference_mode():
            if self.mos_meter.cuda_flag:
                x = x.cuda()
            res = self.mos_meter.forward(x).mean()
        
        return res.cpu().item()
