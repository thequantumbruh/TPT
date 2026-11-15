from datasets import load_dataset
from datasets import interleave_datasets
from transformers import AutoTokenizer


old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(full_ds), vocab_size= 16384) #funny me. Passed the streamed data instead of the corpus
new_tokenizer.save_pretrained("/content/drive/MyDrive/Temi_pretrained_model_tknzr")


T_tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/Temi_pretrained_model_tknzr')
def tokenizer_function (batch):
    return T_tokenizer(batch['text'], add_special_tokens= True)

clean_fineweb = data_set.remove_columns([col for col in data_set.features if col != 'text'])
clean_wiki = factual_dataset.remove_columns([col for col in factual_dataset.features if col != 'text'])


full_ds = interleave_datasets([clean_fineweb, clean_wiki], probabilities=[0.65, 0.35])




train_ds = full_ds.shuffle(buffer_size= 100000, seed = 11)
test_ds = full_ds.shuffle(buffer_size= 100000, seed = 121)
val_ds = full_ds.shuffle(buffer_size= 100000, seed = 42)
train_ds = train_ds.map(tokenizer_function, batch_size= 1000, batched=True)
test_ds = train_ds.map(tokenizer_function, batch_size= 1000, batched=True)
val_ds = val_ds.map(tokenizer_function, batch_size= 1000, batched=True)


def get_data_fast(data_stream, max_token: int, file_name: str):
  buffer = []
  token_count = 0
  bar = tqdm(total=max_token, unit="tok", unit_scale=True, desc=file_name.split("/")[-1])
  with open(file_name, "wb") as file:
    for batch in data_stream:
      input_ids = batch['input_ids']
      if isinstance(input_ids[0], list):
        for text in input_ids:
          buffer.extend(text)
      else:
        buffer.extend(input_ids)

      n = (len(buffer) // 1024) * 1024        # all full 1024-token chunks at once
      if n:
        file.write(np.array(buffer[:n], dtype=np.uint16).tobytes())
        file.flush()
        buffer = buffer[n:]
        token_count += n
        bar.update(n)
        bar.set_postfix(left_in_buffer=len(buffer))

      if token_count >= max_token:
        break
  bar.close()
  print(f"data_saved {token_count} tokens at {file_name}")
  return token_count