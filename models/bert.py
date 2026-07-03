from transformers import AutoModelForSequenceClassification

def get_bert(model_name: str, num_labels: int = 2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model


