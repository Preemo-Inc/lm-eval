import json
import subprocess
from typing import Union

import torch


try:
    from transformers_neuronx.sampling import (
        top_k_top_p_filtering,
        validate_top_k_top_p_min_tokens_to_keep,
    )
except ImportError:
    pass


def get_nc_count() -> Union[int, None]:
    """Returns the number of neuron cores on the current instance."""
    try:
        cmd = "neuron-ls --json-output"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        print(f"inferring nc_count from `neuron-ls` {result.stdout}")
        json_output = json.loads(result.stdout)
        count = sum([x["nc_count"] for x in json_output])
        print(f"nc_count={count}")
        return count
    except Exception:
        return None


def wrap_constant_batch_size(func):
    def _decorator(self, input_ids):
        """input_ids a 2D array with batch_size on dim=0

        makes sure the func runs with self.batch_size
        """
        # access a from TestSample
        batch_size = input_ids.shape[0]

        if batch_size < self.batch_size:
            # handle the event of input_ids.shape[0] != batch_size
            # Neuron cores expect constant batch_size
            input_ids = torch.concat(
                (
                    input_ids,
                    # add missing_batch_size dummy
                    torch.zeros(
                        [self.batch_size - batch_size, *input_ids.size()[1:]],
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


def sample_loop_llama(
    model,
    input_ids,
    start_ids,
    next_token_scores,
    sequence_length,
    eos_token_id=2,
    top_k=50,
    top_p=1.0,
    temperature=1.0,
    streamer=None,
    stopping_criteria_list=None,
):
    validate_top_k_top_p_min_tokens_to_keep(top_k, top_p, None)

    if not isinstance(temperature, float) or not (temperature > 0):
        raise ValueError("temperature has to be a strictly positive float.")

    # stopping_criteria_list = stopping_criteria_list if stopping_criteria_list is not None else StoppingCriteriaList()

    # Flags, one per sequence in a batch, to indicate if a sequence hit eos_token_id
    done_flags = torch.full((input_ids.size(dim=0), 1), False)
    tokens = [input_ids]
    _, start = input_ids.shape

    for cur_len in range(start, sequence_length):
        next_len = cur_len + 1

        if temperature != 1.0:
            next_token_scores /= temperature

        top_values, top_indices = top_k_top_p_filtering(
            next_token_scores, top_k=top_k, top_p=top_p
        )

        # sample
        probs = torch.nn.functional.softmax(top_values, dim=-1, dtype=torch.float32)
        inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
        inputs = torch.gather(top_indices, 1, inputs_in_topk)

        # Update done flags.
        done_flags = torch.logical_or(done_flags, inputs == eos_token_id)
        # Update token id to be eos_token_id if the corresponding done flag is True. For a batch,
        # this means that, while every sequence in the batch has the same length, a sequence that
        # encounters eos_token_id earlier will be filled with eos_token_ids post the first appearance
        # of eos_token_id.

        token = torch.where(done_flags.eq(True), eos_token_id, inputs)
        tokens.append(token)

        if (
            streamer is not None
            and hasattr(streamer, "response_with_prefix")
            and streamer.response_with_prefix
        ):
            streamer.put(torch.cat(tokens, dim=-1))
        elif streamer:
            streamer.put(token)

        if next_len >= sequence_length or done_flags.all():
            break

        if stopping_criteria_list is not None and stopping_criteria_list(
            torch.cat(tokens[-64:], dim=-1), probs
        ):
            break

        # forward pass to get next token
        cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
        next_token_scores = model(inputs, cache_ids, start_ids)

    if streamer:
        streamer.end()

    return torch.cat(tokens, dim=-1)


if __name__ == "__main__":

    class Tester:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        @wrap_constant_batch_size
        def test_constant_batch_size(self, inputs):
            assert len(inputs) == self.batch_size
            return inputs

    batch_size_test = 8
    for i in range(1, batch_size_test + 1):
        tensor = torch.ones([i, 2, 2])
        out = Tester(batch_size=batch_size_test).test_constant_batch_size(tensor)
        torch.testing.assert_allclose(out, tensor)

    try:
        Tester(batch_size=batch_size_test).test_constant_batch_size(
            torch.ones([batch_size_test + 1, 2, 2])
        )
        raise AssertionError("should have raised ValueError")
    except ValueError:
        print("all tests passed")

    print("neuron core count:", get_nc_count())
