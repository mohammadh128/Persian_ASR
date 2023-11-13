# Persian_ASR
A comprehensive Persian Automatic Speech Recognition (ASR) system for accurate transcription and speech-to-text conversion.

## Install Requirements

```bash
pip install -r requirements.txt
```


## Demo

Example code to predict audio.

```bash
python runner.py
```



Example code to predict audio.

```bash
Enter audio path:audio.ogg

امکانات فنی پیاده سازی یک دستیار هوشمند بر پایه هوش مصنوعی
```
## Use in your code

To use ASR in your code

```python
import ASR

prediction = ASR()

result = prediction.predict("audio.ogg")

print(result)

>> امکانات فنی پیاده سازی یک دستیار هوشمند بر پایه هوش مصنوعی

```