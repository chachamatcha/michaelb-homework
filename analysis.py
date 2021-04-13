from transformers import AutoTokenizer
import pandas as pd 

model_checkpoint = "distilbert-base-uncased"
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

df = pd.read_csv("clean.csv")
def f(x):
    return tokenizer(df["headline"],df["summary"])
#embed_df = df.apply(f,axis = 1)

embed_df = tokenizer(list(df["headline"]),list(df["summary"]))