import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)

# Split long text into chunks to avoid memory issues
texts = [
    """Most sleep experts advise that adults get seven to nine hours of sleep per night for good health and emotional well-being, although that changes as you get older. And studies warn that sleeping for less than seven hours a day can increase the risk of obesity, high blood pressure, heart disease and other issues that come with sleep deprivation. CNN has reported on those risks, too, including how getting five hours or less of sleep can increase the likelihood of developing chronic disease.""",

    """Dr. Tony Cunningham, clinical psychologist and director of the Center for Sleep and Cognition in Boston, said it isn't as simple as getting the "right" number of hours of sleep. In a conversation with CNN, he explained his thinking. Sleep quality is just as important as sleep time. A lot of people tend to focus on how many hours of sleep they're getting but neglect the quality of their sleep, which can be even more important than sleep time.""",

    """"There's two different things going on in our bodies, both the type of sleep that we're getting and the quality of sleep that we're getting, and that is our sleep pressure and circadian rhythms," said Cunningham, who is also an assistant professor of psychology at Harvard Medical School. Sleep pressure or sleep drive builds up the longer that you're awake and decreases while you're asleep. It's what causes you to start feeling tired after being awake for an extended period.""",

    """"It's just like eating," Cunningham said. "The longer it's been since you've eaten, the hungrier you get." To get a good night's worth of sleep, you want to get into bed when you've built up a lot of sleep pressure. Your circadian rhythm is your body's internal clock. Don't let the name fool you though. Although external factors, such as light, can affect your circadian rhythm, the pattern that your body follows is guided by your brain.""",

    """"The circadian rhythm can fluctuate and send either sleep-promoting signals or wake-promoting signals," Cunningham said. "So, if you've ever pulled an all-nighter and you've gotten a second wind in the middle of the night, and you felt less tired, then that is your circadian rhythm kicking in." For higher sleep quality, sleep pressure and your circadian rhythm should be working together. That means that any abrupt changes or an irregular sleep schedule can affect your ability to sleep and lower sleep quality.""",

    """One way to enhance sleep quality "is to start waking up at the same time every day, as it can be a little bit more impactful than going to bed at the same time every day — because it's not always a good idea to go to bed if you're not sleepy yet," Cunningham said. Once you have a general sleep schedule or routine, then your body will naturally start to seek out its optimal sleep time.""",
]

# Generate each chunk and concatenate
import gc
all_wavs = []
for i, text in enumerate(texts):
    print(f"Generating chunk {i+1}/{len(texts)}...")
    wav = model.generate(text)
    all_wavs.append(wav)
    gc.collect()  # Free memory between chunks

# Concatenate all audio
final_wav = torch.cat(all_wavs, dim=1)
ta.save("sleep-article.wav", final_wav, model.sr)
print(f"Saved sleep-article.wav")
