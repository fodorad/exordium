import transformers as tfm

Wav2Vec2Processor = tfm.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
Wav2Vec2Model = tfm.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def extract_wav2vec2(speech, sampling_rate: int = 16000):
    input_values = Wav2Vec2Processor(speech, return_tensors="pt", sampling_rate=sampling_rate).input_values  # Batch size 1
    return Wav2Vec2Model(input_values).last_hidden_state
