import os
import re
import hazm
import string
import torch
import warnings
import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
from torch.utils.data import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets, load_from_disk


class ASR:
    _normalizer = hazm.Normalizer()
    chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬",'ٔ', ",", "?", 
    ".", "!", "-", ";", ":",'"',"“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š',
    #     "ء",
    ]
    chars_to_mapping = {
    'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
    'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
    "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
    "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
    'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
    'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",
        
    # "ها": "  ها", "ئ": "ی",
    "۱۴ام": "۱۴ ام",
        
    "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
    "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
    "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
    "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
    "y": " وای ", "z": " زد ",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",
    }
    
    def __pre_load(self):
        metric = evaluate.load("wer")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    def __multiple_replace(self,text, chars_to_mapping):
        pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
        return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

    def __remove_special_characters(self,text, chars_to_ignore_regex):
        text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
        return text

    def __normalizer(self,row, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping):

        text = row['sentence']
        chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
        text = text.lower().strip()

        text = self._normalizer.normalize(text)
        text = self.__multiple_replace(text, chars_to_mapping)
        text = self.__remove_special_characters(text, chars_to_ignore_regex)
        text = re.sub(" +", " ", text)
        _text = []
        for word in text.split():
            try:
                word = int(word)
                _text.append(words(word))
            except:
                _text.append(word)
                
        text = " ".join(_text) + " "
        text = text.strip()

        if not len(text) > 0:
            return None
        
        row['sentence'] = text
        return row

    def __init__(self):
        self.pipe_whisper_small_persian = pipeline("automatic-speech-recognition", "mohammadh128/whisper_small-fa_v03", tokenizer="openai/whisper-small")

    def predict(self,audio_path):
        return self.pipe_whisper_small_persian(audio_path)['text']