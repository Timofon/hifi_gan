from pathlib import Path
from tqdm import tqdm
import torchaudio
from speechbrain.inference.TTS import Tacotron2

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
        data = []

        for path in tqdm(list(Path(audio_dir).iterdir()), desc="Processing audio files"):
            entry = {}
            if path.suffix in [".txt"]:
                entry["path"] = str(path)
                entry["spectrogram"] = self.tacotron2.encode_text(entry["path"])
                
                data.append(entry)
        
        super().__init__(data, *args, **kwargs)
