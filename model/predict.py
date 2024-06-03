from model.classifier import SimpleGPT2SequenceClassifier
from transformers import GPT2Tokenizer
import torch

labels_map = {
    0: "politics",
    1: "sport",
    2: "technology",
    3: "entertainment",
    4: "business"
}
def predict(text):
    model_new = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
    model_new.load_state_dict(torch.load("model/saved_model/flamecat-model.pt"))
    model_new.eval()

    fixed_text = " ".join(text.lower().split())

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model_input = tokenizer(fixed_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    mask = model_input['attention_mask'].cpu()
    input_id = model_input["input_ids"].squeeze(1).cpu()

    output = model_new(input_id, mask)

    pred_label = labels_map[output.argmax(dim=1).item()]
    return pred_label