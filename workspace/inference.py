from transformers import pipeline

PATH_OUTDIR_SENTENCES = ""
NUM_SENTENCES = 5

quest_generator = pipeline('text-generation', model='./gpt2-quest-generator', tokenizer='gpt2', max_length=10000, temperature=0.5)

sentences = []

for i in range(NUM_SENTENCES):
    speech = quest_generator("<SOS>", max_length=100, min_length=10, do_sample=True, temperature=1.2)[0]['generated_text']
    sentences.append(speech)
    if '<EOS>' in speech:
        break

with open(PATH_OUTDIR_SENTENCES, 'w', encoding='utf-8') as out_file:
    for sentence in sentences:
        if sentence[:-1] != '\n':
            sentence += '\n'
        out_file.write(sentence)