"""Phage dataset .

Hosted on HuggingFace at leannmlindsey/PD-GB
"""

from pathlib import Path
from datasets import load_dataset
import torch

from src.dataloaders.utils.rc import coin_flip, string_reverse_complement


class PhageDataset(torch.utils.data.Dataset):
    """
    loads the correct dataset from hugginface
    """

    def __init__(
            self,
            split,
            max_length,
            dataset_name="phage_fragment_inphared",
            d_output=2,  # default binary classification
            dest_path=None,
            tokenizer=None,
            tokenizer_name=None,
            use_padding=None,
            add_eos=False,
            rc_aug=False,
            conjoin_train=False,
            conjoin_test=False,
            return_augs=False,
            return_mask=False,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        assert not (conjoin_train and conjoin_test), "conjoin_train and conjoin_test cannot both be True"
        if (conjoin_train or conjoin_test) and rc_aug:
            print("When using conjoin, we turn off rc_aug.")
            rc_aug = False
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.return_mask = return_mask

        dataset = load_dataset("leannmlindsey/PD-GB", name=dataset_name)

        # Get the correct split
        split_dataset = dataset[split]

        self.split = split

        self.all_seqs = []
        self.all_labels = []

        for entry in split_dataset:
            self.all_seqs.append(entry['sequence'])
            self.all_labels.append(entry['label'])

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        y = self.all_labels[idx]

        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(
            x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq_ids.append(self.tokenizer.sep_token_id)

        if self.conjoin_train or (self.conjoin_test and self.split != "train"):
            x_rc = string_reverse_complement(x)
            seq_rc = self.tokenizer(
                x_rc,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )
            seq_rc_ids = seq_rc["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq_rc_ids.append(self.tokenizer.sep_token_id)
            seq_ids = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)

        else:
            # convert to tensor
            seq_ids = torch.LongTensor(seq_ids)

        # need to wrap in list
        target = torch.LongTensor([y])

        # `seq` has shape:
        #     - (seq_len,) if not conjoining
        #     - (seq_len, 2) for conjoining
        if self.return_mask:
            return seq_ids, target, {"mask": torch.BoolTensor(seq["attention_mask"])}
        else:
            return seq_ids, target

