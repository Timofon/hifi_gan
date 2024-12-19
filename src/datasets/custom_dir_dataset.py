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
            entry["path"] = str(path)
            if path.suffix in [".txt"]:
                entry["spectrogram"] = self.tacotron2.encode_text(entry["path"])
            
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:                
                audio_info = torchaudio.info(entry["path"])
                entry["audio_len"] = audio_info.num_frames / audio_info.sample_rate
            
            data.append(entry)

        print(f'{data=}')

        super().__init__(data, *args, **kwargs)
