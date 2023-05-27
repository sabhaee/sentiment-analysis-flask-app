from transformers import BertModel, BertTokenizer
import torch
# from google.colab import files
import numpy as np
import pandas as pd
import re
from torch import nn
import torch.nn.functional as F
import emoji




random_seed = 62
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device =torch.device("cpu")

MODEL_NAME = "best_model_state_e1.pth"
BERT_model_type = 'bert-base-cased'
# Define the sentiment names
sentiment_names = ['NEGATIVE', 'POSITIVE','NEUTRAL']

# Class to perform sentiment classification
class SentimentClassifier(nn.Module):

  def __init__(self, n_classes,BERT_model_type):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(BERT_model_type)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask)
    
    pooled_output = output['pooler_output']
    output = self.drop(pooled_output)
    return self.out(output)
  
def preprocess_text(tweet):
    tweet = emoji.demojize(tweet)
    # remove character that cause new line when converting to csv
    tweet = re.sub(r'\r\r|\r|\n\n|\n', '',tweet)
    # replace links with 'url'
    tweet = re.sub(r'((https?:\/\/)|(www\.))[A-Za-z0-9.\/]+', 'url',  tweet)
    tweet = re.sub(r'[A-Za-z0-9]+.com', 'url',tweet)
    # remove @users
    tweet = re.sub(r'[@][A-Za-z0-9]+', '',tweet)
    # remove non-ascii chars
    tweet = ''.join([w for w in tweet if ord(w)<128])
    # hastags: bert tokenizer handle such tokens
    tweet = tweet.strip()
    return tweet

def load_model():
  # Loading the model and transfering to the GPU 
  model_path = f"./model/{MODEL_NAME}"
  model = SentimentClassifier(len(sentiment_names),BERT_model_type)
  model.load_state_dict(torch.load(model_path,map_location=device))
  model = model.to(device)

  return model

# Function to get prediction and probability distribution of each tweet using the pretrained model
def model_predict(text, model):
  tokenizer = BertTokenizer.from_pretrained(BERT_model_type)
  #  setting dropout and batch normalization layers to evaluation mode
  
  model = model.eval()
  
  # input_texts = []
  predictions = []
  prediction_probs = []
  sentiment = []
  # Using the tokenizer to encode the tweet:
  encoded_text= tokenizer.encode_plus( text,
                                       max_length=140,
                                       add_special_tokens=True,
                                       return_token_type_ids=False,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
                                       return_tensors='pt',
                                      )
  # Using model to get the sentiment probability and predition
  input_ids = encoded_text['input_ids'].to(device)
  attention_mask = encoded_text['attention_mask'].to(device)
  with torch.no_grad():
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)

    probabilities = F.softmax(outputs, dim=1)

    max_prob = torch.max(probabilities,dim=1)
    # assigning neutral label to sentiments with less that 0.55 probability of 
    # being positive or negative
    sentiment_pred = torch.where(max_prob[0]<0.55,2,preds)


    # input_texts.extend(text)
    predictions.extend(preds)
    prediction_probs.extend(probabilities)
    sentiment.extend(sentiment_pred)


  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  sentiment = torch.stack(sentiment).cpu()
  pred_dict = {
  'class_names': sentiment_names,
  'values': prediction_probs[0].detach().numpy().tolist()}
  
  return {"sentiment":sentiment_names[sentiment], "score":prediction_probs[0].detach().numpy().tolist()[:-1]}