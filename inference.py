# -*- coding: utf-8 -*-

seed_all()
device = get_device()

from utils import seed_all, get_device
from models import  GetEncodings, SearchSimilar
bart_tokenizer

# Inference
def inference(question, bart_tokenizer, bart_model, df_context, model_op, MODEL_STORE):

    # Get Pretrained BERT encodings
    ge = GetEncodings(MODEL_STORE = MODEL_STORE, type='questions')
    encoded_question = ge.encode(question, max_length=30)

    # Find top matching documents
    ss = SearchSimilar(iterator = df_context['context'].values.tolist(), filename='index.bin', embeddings=model_op, shape=768, device=device)
    similar_contexts = ss.get_n_similar_vectors(encoded_question, 3)
    similar_contexts.insert(0, question)

    combined_tokens = '</s></s>'.join(similar_contexts)

    print(f'Top similar document outputs is {combined_tokens}')

    # Prepare data for BART Inferencing

    source_encoding = bart_tokenizer(
            combined_tokens,
            max_length=1024,
            padding='max_length',
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt")
   

    # Inference BART Model
    output = bart_model(
            source_encoding['input_ids'].to(device),
            mode = 'eval')
    output = bart_tokenizer.decode(output[0])
    print(output)
    return output
