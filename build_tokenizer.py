from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
import configs
import os

tokenizer = Tokenizer(BPE())
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

def main():
    tokenizer.train(trainer, [configs.data.raw1_cut])
    tokenizer.save(os.path.join(configs.data.path, 'bpe.vocab'))
    print(f"save to {configs.data.path}")


