from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
with open('10K.txt') as f:
    text = f.read()

summary = summarizer(text[:1024], max_length=200, min_length=50, do_sample=False)
print(summary[0]['summary_text'])
