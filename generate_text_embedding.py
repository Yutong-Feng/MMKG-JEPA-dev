import torch
import argparse
import os
from transformers import AutoModel, LlamaModel


def extract_text_embeddings(model_name: str, save_path: str):
    """
    Extract word embeddings from Hugging Face text models.
    """
    print(f"Loading text model: {model_name}...")

    if "llama" in model_name.lower():
        model = LlamaModel.from_pretrained(model_name)
        # LLaMA's embedding layer
        embeddings = model.embed_tokens.weight.detach().cpu()
    else:
        model = AutoModel.from_pretrained(model_name)
        # BERT / RoBERTa's embedding layer
        embeddings = model.embeddings.word_embeddings.weight.detach().cpu()

    print(f"Text embedding shape: {embeddings.shape}")
    torch.save(embeddings, save_path)
    print(f"Saved textual embeddings to {save_path}")


if __name__ == "__main__":
    extract_text_embeddings(
        model_name="bert-base-uncased",
        save_path="tokens/textual.pth",
    )
