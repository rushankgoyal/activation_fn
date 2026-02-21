"""Dataset loading utilities for Booksum and WikiText-103.

Replicates the data pipeline used in:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

The paper trains on two datasets:
  - Booksum Complete Cleaned (kmfoda/booksum): 145K train / 24K test sequences
  - WikiText-103 (wikitext-103-raw-v1):         510K train / 1.2K test sequences

Both datasets are tokenised with the GPT-2 BPE tokenizer (vocab size 50,257)
and chunked into fixed-length sequences of `max_seq_len` tokens for language
modelling.  The training objective is next-token prediction evaluated with
mean cross-entropy loss.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Default tokenizer: GPT-2 BPE (vocab size 50,257)
DEFAULT_TOKENIZER = "gpt2"


class TextDataset(Dataset):
    """Fixed-length token chunks from a raw text corpus.

    Concatenates all text in the corpus, tokenises it, and splits the resulting
    token sequence into non-overlapping windows of `max_seq_len` tokens.  Each
    window provides an (input, target) pair where target is the input shifted
    one position to the right (standard next-token prediction).

    Args:
        token_ids: 1-D tensor of all token ids for the split.
        max_seq_len: Context length (number of tokens per sample).
    """

    def __init__(self, token_ids: torch.Tensor, max_seq_len: int = 1024) -> None:
        self.max_seq_len = max_seq_len
        # Trim so we have complete chunks of length max_seq_len + 1
        n_chunks = (len(token_ids) - 1) // max_seq_len
        self.data = token_ids[: n_chunks * max_seq_len + 1]

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return x, y


def _tokenize_corpus(
    texts: list[str],
    tokenizer,
    max_seq_len: int,
    text_field: str = "text",
) -> torch.Tensor:
    """Tokenise a list of strings and concatenate into one long token tensor."""
    all_ids: list[int] = []
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)  # sentence separator

    return torch.tensor(all_ids, dtype=torch.long)


def load_booksum(
    split: str = "train",
    max_seq_len: int = 1024,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    cache_dir: Optional[str] = None,
) -> TextDataset:
    """Load the Booksum Complete Cleaned dataset.

    The paper uses the `kmfoda/booksum` dataset from the Hugging Face Hub.
    The corpus is concatenated, tokenised, and chunked into fixed-length windows.

    Args:
        split: Dataset split ('train' or 'test').
        max_seq_len: Context window size in tokens.
        tokenizer_name: HuggingFace tokenizer to use (default: 'gpt2').
        cache_dir: Optional local cache directory for the dataset.

    Returns:
        A TextDataset ready for use with a DataLoader.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from e

    logger.info(f"Loading Booksum ({split} split)...")
    ds = load_dataset("kmfoda/booksum", split=split, cache_dir=cache_dir)

    # The dataset has a 'chapter' field containing the raw text
    text_field = "chapter" if "chapter" in ds.column_names else ds.column_names[0]
    texts = [str(row[text_field]) for row in ds]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    token_ids = _tokenize_corpus(texts, tokenizer, max_seq_len)
    logger.info(f"  Booksum {split}: {len(token_ids):,} tokens → {len(token_ids) // max_seq_len:,} sequences")
    return TextDataset(token_ids, max_seq_len)


def load_wikitext(
    split: str = "train",
    max_seq_len: int = 1024,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    cache_dir: Optional[str] = None,
) -> TextDataset:
    """Load the WikiText-103 dataset.

    Uses the `wikitext-103-raw-v1` configuration from the Hugging Face Hub.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        max_seq_len: Context window size in tokens.
        tokenizer_name: HuggingFace tokenizer to use (default: 'gpt2').
        cache_dir: Optional local cache directory for the dataset.

    Returns:
        A TextDataset ready for use with a DataLoader.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from e

    logger.info(f"Loading WikiText-103 ({split} split)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
    texts = [row["text"] for row in ds]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    token_ids = _tokenize_corpus(texts, tokenizer, max_seq_len)
    logger.info(f"  WikiText-103 {split}: {len(token_ids):,} tokens → {len(token_ids) // max_seq_len:,} sequences")
    return TextDataset(token_ids, max_seq_len)


def make_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Wrap a TextDataset in a PyTorch DataLoader.

    Args:
        dataset: A TextDataset instance.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle the data each epoch.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        A DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
