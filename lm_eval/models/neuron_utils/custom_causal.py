import torch
from typing import Optional
import copy
from transformers import GenerationConfig, PretrainedConfig
from transformers.generation import StoppingCriteriaList
import logging

try:
    NEURON_AVAILABLE = True
    from optimum.neuron import NeuronModelForCausalLM
    from optimum.neuron.generation import TokenSelector
except ImportError:
    NeuronModelForCausalLM = object
    NEURON_AVAILABLE = False

logger = logging.getLogger(__name__)


def wrap_constant_batch_size(func):
    def _decorator(self, input_ids):
        """input_ids a 2D array with batch_size on dim=0

        makes sure the func runs with self.batch_size
        """
        # access a from TestSample
        batch_size = input_ids.shape[0]

        if self.batch_size < batch_size:
            # handle the event of input_ids.shape[0] != batch_size
            # Neuron cores expect constant batch_size
            input_ids = torch.concat(
                (
                    input_ids,
                    # add missing_batch_size dummy
                    torch.zeros(
                        [batch_size - self.batch_size, *input_ids.size()[1:]],
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ),
                dim=0,
            )
        elif batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        # return the forward pass that requires constant batch size
        return func(self, input_ids)[:batch_size]

    return _decorator


class CustomNeuronModelForCausalLM(NeuronModelForCausalLM):
    @wrap_constant_batch_size
    def forward_full_sequence(self, input_ids: torch.Tensor):
        """
        get logits for the entire sequence

        :param input_ids: torch.Tensor
            A torch tensor of shape [batch, sequence_cont]
            the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
            logits returned from the model's decoder-lm head
        """
        _, sequence_length = input_ids.shape

        with torch.inference_mode():
            cache_ids = torch.arange(0, sequence_length, dtype=torch.int32).split(1)
            input_ids_split = input_ids.split(1, dim=1)

            return torch.concat(
                [
                    self.forward(
                        input_ids=input_id, cache_ids=cache_id, return_dict=False
                    )[0]
                    for input_id, cache_id in zip(input_ids_split, cache_ids)
                ],
                dim=1,
            )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        generation_config: Optional["GenerationConfig"] = None,
        **kwargs,
    ) -> torch.LongTensor:
        r"""
        A streamlined generate() method overriding the transformers.GenerationMixin.generate() method.

        This method uses the same logits processors/warpers and stopping criteria as the transformers library
        `generate()` method but restricts the generation to greedy search and sampling.

        It does not support transformers `generate()` advanced options.

        Please refer to https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        for details on generation configuration.

        Parameters:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~transformers.generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.

        Returns:
            `torch.Tensor`: A  `torch.FloatTensor`.
        """
        # The actual generation configuration is a combination of config and parameters
        generation_config = copy.deepcopy(
            self.generation_config if generation_config is None else generation_config
        )
        model_kwargs = generation_config.update(
            **kwargs
        )  # All unused kwargs must be model kwargs
        # Check model kwargs are actually used by either prepare_inputs_for_generation or forward
        self._validate_model_kwargs(model_kwargs)

        # Instantiate a TokenSelector for the specified configuration
        selector = TokenSelector.create(
            input_ids, generation_config, self, self.max_length
        )
        # MICHAELS CHANGE: PATCH IN stopping_criteria
        selector.stopping_criteria.append(stopping_criteria)
        # Verify that the inputs are compatible with the model static input dimensions
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        padded_input_ids = input_ids
        padded_attention_mask = attention_mask
        if batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        elif batch_size < self.batch_size:
            logger.warning(
                "Inputs will be padded to match the model static batch size. This will increase latency."
            )
            padding_shape = [self.batch_size - batch_size, sequence_length]
            padding = torch.full(
                padding_shape, fill_value=self.config.eos_token_id, dtype=torch.int64
            )
            padded_input_ids = torch.cat([input_ids, padding])
            if attention_mask is not None:
                padding = torch.zeros(padding_shape, dtype=torch.int64)
                padded_attention_mask = torch.cat([attention_mask, padding])
        # Drop the current generation context and clear the Key/Value cache
        self.reset_generation()

        output_ids = self.generate_tokens(
            padded_input_ids,
            selector,
            batch_size,
            attention_mask=padded_attention_mask,
            **model_kwargs,
        )
        return output_ids[:batch_size, :]
