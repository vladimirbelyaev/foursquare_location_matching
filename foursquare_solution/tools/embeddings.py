import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def load_model(name, use_cuda=True):
    print('loading model and tokenizer')
    model = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    print('sending model to cuda')
    if use_cuda:
        model = model.to('cuda:0')
    return model, tokenizer



def convert_to_embeds(
    sentences,
    model_tok,
    batch_size=512,
    progress_bar=True,
    use_cuda=True
):
    # Load model from HuggingFace Hub
    if isinstance(model_tok, str):
        model, tokenizer = load_model(model_tok, use_cuda)
    else:
        model, tokenizer = model_tok

    num_batches = (len(sentences) - 1) // batch_size + 1
    embeds = []
    print('start getting embeds')
    rng = range(num_batches)
    if progress_bar:
        rng = tqdm(rng)
    for batch_idx in rng:
    # Tokenize sentences
        encoded_input = tokenizer(
            sentences[batch_idx * batch_size: (batch_idx + 1) * batch_size],
            padding=True, truncation=True, return_tensors='pt'
        )
        if use_cuda:
            encoded_input = encoded_input.to('cuda:0')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask']).cpu()
        del encoded_input, model_output
        embeds.append(sentence_embeddings)
    if isinstance(model_tok, str):
        del model, tokenizer
    if use_cuda:
        torch.cuda.empty_cache()
    return torch.vstack(embeds).numpy()

