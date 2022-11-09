import transformers as tfm


class Wav2vec2Wrapper():

    def __init__(self) -> None:
        self.preprocessor = tfm.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = tfm.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


    def extract_wav2vec2(self, speech, sampling_rate: int = 16000):
        input_values = self.preprocessor(speech, return_tensors="pt", sampling_rate=sampling_rate).input_values  # Batch size 1
        return self.model(input_values).last_hidden_state
