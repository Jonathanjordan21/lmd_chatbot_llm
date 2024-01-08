from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_asr_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    return model, processor


def audio2text(model, processor, sample, sampling_rate=22050):
    input_features = processor(sample, sampling_rate=sampling_rate, return_tensors="pt").input_features 

    predicted_ids = model.generate(input_features)
    
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]