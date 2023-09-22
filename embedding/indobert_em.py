import ctranslate2
import transformers
import numpy as np
import torch

model_path = "D:\kompas-dev\\ai\models\indobert-base-uncased-ct2"
model_name = "indolem/indobert-base-uncased"
device = "cpu"

def embedder(text: str):

    encoder = ctranslate2.Encoder(model_path, device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inputs = [text]

    tokens = tokenizer(inputs).input_ids

    output = encoder.forward_batch(tokens)
    pooler_output = output.pooler_output

    pooler_output = np.array(pooler_output)
    pooler_output = torch.as_tensor(pooler_output)

    formatted_vector = [float(val.item()) for val in pooler_output[0]]

    return formatted_vector