from pathlib import Path
from tqdm import tqdm
import torchaudio

from src.datasets.base_dataset import BaseDataset


class LJSpeechDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in tqdm(list(Path(audio_dir).iterdir()), desc="Processing audio files"):
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                
                audio_info = torchaudio.info(entry["path"])
                entry["audio_len"] = audio_info.num_frames / audio_info.sample_rate
                data.append(entry)
        
        super().__init__(data, *args, **kwargs)
