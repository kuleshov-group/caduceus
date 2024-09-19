"""Genomic Benchmarks Dataset.

From: https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
"""

from pathlib import Path

import torch
from genomic_benchmarks.data_check import is_downloaded
from genomic_benchmarks.loc2seq import download_dataset

from src.dataloaders.utils.rc import coin_flip, string_reverse_complement


class GenomicBenchmarkDataset(torch.utils.data.Dataset):
    """
    Loop through bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    """

    def __init__(
            self,
            split,
            max_length,
            dataset_name="human_nontata_promoters",
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

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        self.split = split

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_seqs = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for path in (base_path / label_type).iterdir():
                with open(path, "r") as f:
                    content = f.read()
                self.all_seqs.append(content)
                self.all_labels.append(label_mapper[label_type])

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

class GenomeEvaluationBenchmark(HG38):
    _name_ = "gue"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(
            self, dataset_name, train_val_split_seed,
            dest_path=None, tokenizer_name="char", d_output=None, rc_aug=False,
            conjoin_train=False, conjoin_test=False,
            max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
            padding_side="left", val_ratio=0.0005, val_split_seed=2357, add_eos=False,
            detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
            shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
            fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs
    ):

        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == "bpe":
            logger.info("**Using BPE tokenizer**")
            #self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained("leannmlindsey/mamba_hg38_BPE_ntp_rc_aug_seqlen-4k_d_model-256_n_layer-4_lr-8e-6", trust_remote_code=True)
            if hasattr(self, 'max_length'):
                self.tokenizer.model_max_length = self.max_length
                self.tokenizer.init_kwargs['model_max_length'] = self.max_length
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        # Create all splits: torch datasets (only train/test in this benchmark, val created below)
        self.dataset_train, self.dataset_test = [
            GueDataset(
                split=split,
                max_length=max_len,
                dataset_name=self.dataset_name,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                dest_path=self.dest_path,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        val_data, train_data = torch.utils.data.random_split(
            list(zip(self.dataset_train.all_seqs, self.dataset_train.all_labels)),
            lengths=[0.1, 0.9],
            generator=torch.Generator().manual_seed(self.train_val_split_seed)
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.all_seqs = [train_data[i][0] for i in range(len(train_data))]
        self.dataset_train.all_labels = [train_data[i][1] for i in range(len(train_data))]

        self.dataset_val.all_seqs = [val_data[i][0] for i in range(len(val_data))]
        self.dataset_val.all_labels = [val_data[i][1] for i in range(len(val_data))]
        self.dataset_val.split = "val"

class PhageDetectionGenomicBenchmark(HG38):
    _name_ = "phage"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(
            self, dataset_name, train_val_split_seed,
            dest_path=None, tokenizer_name="char", d_output=None, rc_aug=False,
            conjoin_train=False, conjoin_test=False,
            max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
            padding_side="left", val_ratio=0.0005, val_split_seed=2357, add_eos=False,
            detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
            shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
            fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs
    ):

        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        elif self.tokenizer_name == "bpe":
            logger.info("**Using BPE tokenizer**")
            #self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained("leannmlindsey/mamba_hg38_BPE_ntp_rc_aug_seqlen-4k_d_model-256_n_layer-4_lr-8e-6", trust_remote_code=True)
            if hasattr(self, 'max_length'):
                self.tokenizer.model_max_length = self.max_length
                self.tokenizer.init_kwargs['model_max_length'] = self.max_length
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        # Create all splits: torch datasets (only train/test in this benchmark, val created below)
        self.dataset_train, self.dataset_test = [
            PhageDataset(
                split=split,
                max_length=max_len,
                dataset_name=self.dataset_name,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                dest_path=self.dest_path,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        val_data, train_data = torch.utils.data.random_split(
            list(zip(self.dataset_train.all_seqs, self.dataset_train.all_labels)),
            lengths=[0.1, 0.9],
            generator=torch.Generator().manual_seed(self.train_val_split_seed)
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.all_seqs = [train_data[i][0] for i in range(len(train_data))]
        self.dataset_train.all_labels = [train_data[i][1] for i in range(len(train_data))]

        self.dataset_val.all_seqs = [val_data[i][0] for i in range(len(val_data))]
        self.dataset_val.all_labels = [val_data[i][1] for i in range(len(val_data))]
        self.dataset_val.split = "val"



