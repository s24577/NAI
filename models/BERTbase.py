from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("SharanSMenon/22-languages-bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("SharanSMenon/22-languages-bert-base-cased")


def predict(sentence):
    tokenized = tokenizer(sentence, return_tensors="pt")
    outputs = model(**tokenized)
    return model.config.id2label[outputs.logits.argmax(dim=1).item()]


sentence1 = "in war resolution, in defeat defiance, in victory magnanimity"
predict(sentence1)
print(predict(sentence1))

sentence2 = "en la guerra resolución en la derrota desafío en la victoria magnanimidad"
predict(sentence2)
print(predict(sentence2))

sentence3 = "هذا هو أعظم إله على الإطلاق"
predict(sentence3)
print(predict(sentence3))
