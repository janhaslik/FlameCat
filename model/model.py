import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from transformers import GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
from model.dataset import Dataset
from model.classifier import SimpleGPT2SequenceClassifier

data_file = 'dataset.csv'
model_params = {
    'hidden_size': 768,
    'num_classes': 5,
    'max_seq_len': 128,
    'gpt_model_name': "gpt2"
}
labels = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4
}
class ModelTrainer:
    def __init__(self, tokenizer_name='gpt2'):
        self.data_file = data_file
        self.model_params = model_params
        self.labels = labels
        self.tokenizer_name = tokenizer_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.load_data()
        trained_model = self.train(epochs=1)
        true_labels, pred_labels = self.evaluate(trained_model)
        torch.save(trained_model.state_dict(), "model/saved_model/flamecat-model.pt")

    def load_data(self):
        df = pd.read_csv(self.data_file)
        np.random.seed(112)
        self.df_train, self.df_val, self.df_test = np.split(df.sample(frac=1, random_state=35),
                                                            [int(0.8*len(df)), int(0.2*len(df))])

    def train(self, epochs=1, learning_rate=1e-5):
        model = SimpleGPT2SequenceClassifier(**self.model_params)
        train_data = Dataset(self.df_train, self.labels, self.tokenizer)
        val_data = Dataset(self.df_val, self.labels, self.tokenizer)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        for epoch_num in range(epochs):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input["input_ids"].squeeze(1).to(device)

                model.zero_grad()

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                """
                print(
                    f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.df_train): .3f} \
                    | Train Accuracy: {total_acc_train / len(self.df_train): .3f} \
                    | Val Loss: {total_loss_val+1 / len(self.df_val+1): .3f} \
                    | Val Accuracy: {total_acc_val+1 / len(self.df_val): .3f}")"""

        return model

    def evaluate(self, model):
        test_data = Dataset(self.df_test, self.labels, self.tokenizer)

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            model = model.cuda()

        # Tracking variables
        predictions_labels = []
        true_labels = []

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc

                # add original labels
                true_labels += test_label.cpu().numpy().flatten().tolist()
                # get predictions to list
                predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

        print(f'Test Accuracy: {total_acc_test / len(self.df_test): .3f}')
        return true_labels, predictions_labels


"""
trainer = ModelTrainer(data_file, model_params, labels)
trainer.load_data()
trained_model = trainer.train(epochs=1)
true_labels, pred_labels = trainer.evaluate(trained_model)
torch.save(trained_model.state_dict(), "model/saved_model/flamecat-model.pt")
"""
