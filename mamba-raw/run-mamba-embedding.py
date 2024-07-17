import argparse
import dataclasses

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
import numpy as np
import torch
import tqdm
from transformers import MambaForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Run Mamba to get dataset embeddings.")
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--dataset-path", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=False)
parser.add_argument("--dataset-split", type=str, required=False, default="validation")
parser.add_argument("--layer-number", type=int, required=False, default=31)
parser.add_argument("--device", type=str, required=False, default="auto")
parser.add_argument("--batch-size", type=int, required=False, default=1)

VALID_MODEL_NAMES = [
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
]

@dataclasses.dataclass
class Args:
    model_name: str
    dataset_path: str
    layer_number: int = None
    dataset_name: str = None
    dataset_split: str = "validation"
    device: str = "auto"
    batch_size: int = 1

    def __post_init__(self) -> None:
        # Check that a valid Mamba model name was passed
        if self.model_name not in VALID_MODEL_NAMES:
            raise ValueError(f"Invalid model name: {self.model_name}. Valid model names are: {VALID_MODEL_NAMES}")
        else:
            self.model = MambaForCausalLM.from_pretrained(self.model_name, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set the layer number to the middle layer if not provided
        if self.layer_number is None:
            self.layer_number = self.model.backbone.layers // 2
        # Verify that the layer number is valid for this Mamba model
        if self.layer_number < 0 or self.layer_number >= len(self.model.backbone.layers):
            raise ValueError(f"Invalid layer number: {self.layer_number}. Valid layer numbers are: 0 to {self.model.backbone.layers - 1}")

        # Check that the dataset path exists
        try:
            self.dataset = load_dataset(self.dataset_path, self.dataset_name)
            # Check that the dataset split is valid
            if self.dataset_split not in self.dataset.keys():
                raise ValueError(f"Invalid dataset split: {self.dataset_split}. Valid dataset splits are: {self.dataset.keys()}")
        except DatasetNotFoundError as e:
            raise ValueError(f"Dataset not found: {self.dataset_path} with name: {self.dataset_name}") from e



def main(args: Args) -> None:
    model = args.model
    tokenizer = args.tokenizer
    dataset = args.dataset[args.dataset_split]

    model.eval()
    embeddings: list[np.ndarray] = [None] * len(dataset)
    batch_idxs = list(range(0, len(dataset), args.batch_size))
    with torch.no_grad():
        for batch_idx in tqdm.tqdm(batch_idxs):
            data = dataset[batch_idx:batch_idx + args.batch_size]
            # TODO: Filter out HTML? Better text parsing?"
            texts = [" ".join(ex["tokens"]["token"]) for ex in data["document"]]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, return_dict=True)
            embedding_batch = outputs.cache_params.ssm_states[args.layer_number].detach().cpu().numpy()
            embeddings[batch_idx:batch_idx + args.batch_size] = embedding_batch
            # Save every 100 examples
            if (batch_idx // args.batch_size) % (100 // args.batch_size) == 0:
                np.savez("embeddings.npz", embeddings=embeddings[:batch_idx + args.batch_size])


if __name__ == "__main__":
    input_args = parser.parse_args()
    args = Args(**vars(input_args))
    main(args)
