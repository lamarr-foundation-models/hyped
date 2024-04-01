import datasets
import ray
from distributed_gpu_processing import (
    ApplyModelProcessor,
    ApplyModelProcessorConfig,
)

from hyped.data.dist.parallel import DistributedParallelDataPipe
from hyped.data.dist.pipe import DistributedDataPipe
from hyped.data.processors.tokenizers.hf import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerConfig,
)

if __name__ == "__main__":
    ray.init()

    # load dataset
    ds = datasets.load_dataset("imdb", split="train")
    # specify pipeline
    pipe = DistributedDataPipe(
        [
            DistributedParallelDataPipe(
                [
                    DistributedDataPipe(
                        [
                            HuggingFaceTokenizer(
                                HuggingFaceTokenizerConfig(
                                    tokenizer="bert-base-uncased",
                                    text="text",
                                    max_length=128,
                                    padding="max_length",
                                    truncation=True,
                                    return_attention_mask=True,
                                )
                            ),
                            ApplyModelProcessor(
                                ApplyModelProcessorConfig(
                                    pretrained_ckpt="bert-base-uncased"
                                )
                            ),
                        ],
                        proc_options=dict(num_gpus=1),
                    ),
                    DistributedDataPipe(
                        [
                            HuggingFaceTokenizer(
                                HuggingFaceTokenizerConfig(
                                    tokenizer="roberta-base",
                                    text="text",
                                    max_length=128,
                                    padding="max_length",
                                    truncation=True,
                                    return_attention_mask=True,
                                )
                            ),
                            ApplyModelProcessor(
                                ApplyModelProcessorConfig(
                                    pretrained_ckpt="roberta-base"
                                )
                            ),
                        ],
                        proc_options=dict(num_gpus=1),
                    ),
                ]
            )
        ],
    )
    # apply pipe to dataset
    ds = pipe.apply(ds, num_proc=1)

    ray.shutdown()
