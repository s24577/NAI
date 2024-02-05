from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("SharanSMenon/22-languages-bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("SharanSMenon/22-languages-bert-base-cased")


def predict(in_sentence):
    tokenized = tokenizer(in_sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokenized)
    return model.config.id2label[outputs.logits.argmax(dim=1).item()]
