import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel
import random
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.data
import nltk

from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_path = "train.csv"
test_path = "test.csv"
test_labels_paths = "test_labels.csv"
test_df = pd.read_csv(test_path)
test_labels_df = pd.read_csv(test_labels_paths)
test_df = pd.concat([test_df.iloc[:, 1], test_labels_df.iloc[:, 1:]], axis = 1)
test_df.to_csv("test-dataset.csv")
test_dataset_path = "test-dataset.csv"

#Lets make a new column labeled "healthy"

def healthy_filter(df):
  if (df["toxic"]==0) and (df["severe_toxic"]==0) and (df["obscene"]==0) and (df["threat"]==0) and (df["insult"]==0) and (df["identity_hate"]==0):
    return 1
  else:
    return 0

attributes = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate', 'healthy']

class Comments_Dataset(Dataset):
  def __init__(self, data_path, tokenizer, attributes, max_token_len = 128, sample=5000):
    self.data_path = data_path
    self.tokenizer = tokenizer
    self.attributes = attributes
    self.max_token_len = max_token_len
    self.sample = sample
    self._prepare_data()

  def _prepare_data(self):
    data = pd.read_csv(self.data_path)
    data["healthy"] = data.apply(healthy_filter,axis=1)
    data["unhealthy"] = np.where(data['healthy']==1, 0, 1)
    if self.sample is not None:
      unhealthy = data.loc[data["healthy"] == 0]
      healthy = data.loc[data["healthy"] ==1]
      self.data = pd.concat([unhealthy, healthy.sample(self.sample, random_state=42)])
    else:
      self.data = data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,index):
    item = self.data.iloc[index]
    comment = str(item.comment_text)
    attributes = torch.FloatTensor(item[self.attributes])
    tokens = self.tokenizer.encode_plus(comment,
                                      add_special_tokens=True,
                                      return_tensors='pt',
                                      truncation=True,
                                      padding='max_length',
                                      max_length=self.max_token_len,
                                      return_attention_mask = True)
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}


class Comments_Data_Module(pl.LightningDataModule):

  def __init__(self, train_path, val_path, attributes, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
    super().__init__()
    self.train_path = train_path
    self.val_path = val_path
    self.attributes = attributes
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    if stage in (None, "fit"):
      self.train_dataset = Comments_Dataset(self.train_path, attributes=self.attributes, tokenizer=self.tokenizer)
      self.val_dataset = Comments_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)
    if stage == 'predict':
      self.val_dataset = Comments_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

comments_data_module = Comments_Data_Module(train_path, test_dataset_path, attributes=attributes)
comments_data_module.setup()
comments_data_module.train_dataloader()

class Comment_Classifier(pl.LightningModule):
#the config dict has the hugginface parameters in it
  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    self.loss_func = nn.CrossEntropyLoss()
    self.dropout = nn.Dropout()
    
  def forward(self, input_ids, attention_mask, labels=None):
    # roberta layer
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits / classification layers
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    # calculate loss
    loss = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
    return loss, logits

  def training_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    self.log("train loss ", loss, prog_bar = True, logger=True)
    return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

  def validation_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    self.log("validation loss ", loss, prog_bar = True, logger=True)
    return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}

  def predict_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    return outputs

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size']
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]


  # def validation_epoch_end(self, outputs):
  #   losses = []
  #   for output in outputs:
  #     loss = output['val_loss'].detach().cpu()
  #     losses.append(loss)
  #   avg_loss = torch.mean(torch.stack(losses))
  #   self.log("avg_val_loss", avg_loss)
    
config = {
    'model_name': 'distilroberta-base',
    'n_labels': len(attributes),
    'batch_size': 128,
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(comments_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 100
}

##tokenizer
model_name = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = Comment_Classifier(config=config)
model.load_state_dict(torch.load("model_state_dict.pt"))
model.eval()

# lemmatizing tool
lemmatizer = WordNetLemmatizer()

def prepare_tokenized_review(raw_review):
  # Remove HTML tags with BS
  review_text = BeautifulSoup(raw_review).get_text()
  # Removing non-letters using a regular expression
  review_text = re.sub("[^a-zA-Z!?]"," ", review_text)
  # Convert words to lower case and split them
  words = review_text.lower().split()
  # Remove stop-words
#   stops = set(stopwords.words("english"))
#   words = [w for w in words if not w in stops]
  # Lemmatize the word list
  lemmatized = []
  for word in words:
    lemmatized.append(lemmatizer.lemmatize(word))
  return " ".join(lemmatized)

def get_encodings(text):
    MAX_LEN=256
    encodings = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt')
    return encodings

def run_inference(encoding):
  with torch.no_grad():
    input_ids = encoding['input_ids'].to(device, dtype=torch.long)
    attention_mask = encoding['attention_mask'].to(device, dtype=torch.long)
    output = model(input_ids, attention_mask)
    final_output = torch.softmax(output[1][0],dim=0).cpu()
    print(final_output.numpy().tolist())
    return final_output.numpy().tolist()

# test_encoding = get_encodings("I think you are wonderful and deserve the best!")
# test_result = run_inference(test_encoding)

test_tweets = test_df["comment_text"].values
#streamlit section
models = ["distilroberta-base"]
model_pointers = ["default: distilroberta-base"]

# current_random_tweet = test_tweets[random.randint(0,len(test_tweets))]
# current_random_tweet = prepare_tokenized_review(current_random_tweet)
st.write("1. Hit the button to view and see the analyis of a random tweet")


# current_random_tweet = test_tweets[random.randint(0,len(test_tweets))]
# current_random_tweet = prepare_tokenized_review(current_random_tweet)
# st.write(current_random_tweet)
# if st.button("New Tweet"):
    # st.experimental_rerun()
    


#######################################

# st.write(device)
# st.write(test_result)
# st.write(attributes[test_result.index(max(test_result))])
with st.form(key="init_form"):
    current_random_tweet = test_tweets[random.randint(0,len(test_tweets))]
    current_random_tweet = prepare_tokenized_review(current_random_tweet)
    # st.write(current_random_tweet)


    choice = st.selectbox("Choose Model", model_pointers)

# The index of choice in model_pointers will access the models list
    # and select the Hugging Face model path at index.  
    user_picked_model = models[model_pointers.index(choice)]
    with st.spinner("Analyzing..."):
        text_encoding = get_encodings(current_random_tweet)
        result = run_inference(text_encoding)
        df = pd.DataFrame({"Tweet":current_random_tweet}, index=[0])
        df["Highest Toxicity Class"] = attributes[result.index(max(result))]
        df["Sentiment Score"] = max(result)
        st.table(df)
        # st.write(input_text)
        # st.write(text_encoding)
        # st.write(result)
        # st.write(attributes[result.index(max(result))])

    next_tweet = st.form_submit_button("Next Tweet")

if next_tweet:
    with st.spinner("Analyzing..."):
        st.write("")
        # text_encoding = get_encodings(current_random_tweet)
        # result = run_inference(text_encoding)
        # # st.write(input_text)
        # # st.write(text_encoding)
        # st.write(result)
        # st.write(attributes[result.index(max(result))])