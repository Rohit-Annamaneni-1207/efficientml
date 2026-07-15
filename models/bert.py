from transformers import AutoModelForSequenceClassification


def get_bert(
    model_name: str,
    num_labels: int | None = None,
):

    kwargs = {}

    if num_labels is not None:
        kwargs["num_labels"] = num_labels

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **kwargs,
    )