import os

import torch
import transformers

from transformers import LlamaForCausalLM
from transformers import OPTForCausalLM
from transformers import MistralForCausalLM
import copy
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import torch.nn.functional as F

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from lm_eval.utils import MultiTokenEOSCriteria, stop_sequences_criteria

from typing import List, Optional, Union
from huggingface_hub.constants import HF_HUB_CACHE
import enum
from dataclasses import dataclass
from typing import Type, Callable

NEURON_AVAILABLE = True
try:
    from .neuron_utils import shared_utils
    from transformers_neuronx.llama.model import LlamaForSampling
    from transformers_neuronx.opt.model import OPTForSampling
    from transformers_neuronx.mistral.model import MistralForSampling
    from transformers_neuronx.module import save_pretrained_split
    from transformers_neuronx import sampling
    from transformers_neuronx.stopping_criteria import StoppingCriteriaList

    # def stop_sequences_criteria(
    #     tokenizer: transformers.PreTrainedTokenizer,
    #     stop_sequences: List[str],
    #     initial_decoder_input_length: int,
    #     batch_size: int,
    # ) -> StoppingCriteriaList:
    #     return StoppingCriteriaList(
    #         [
    #             *[
    #                 MultiTokenEOSCriteria(
    #                     sequence, tokenizer, initial_decoder_input_length, batch_size
    #                 )
    #                 for sequence in stop_sequences
    #             ],
    #         ]
    #     )

    @dataclass
    class ModelRegistry:
        model_class_hf: Union[Type[LlamaForCausalLM], Type[OPTForCausalLM]]
        model_class_neuron: Union[Type[LlamaForSampling], Type[OPTForSampling]]
        sampling_fn: Type[Callable] = sampling.sample_loop_llama

    class ModelTypeToClasses(enum.Enum):
        """
        Maps HuggingFace model types to Neuron model types.
        """

        LlamaForCausalLM = ModelRegistry(
            model_class_hf=LlamaForCausalLM,
            model_class_neuron=LlamaForSampling,
        )
        OPTForCausalLM = ModelRegistry(
            model_class_hf=OPTForCausalLM, model_class_neuron=OPTForSampling
        )
        MistralForCausalLM = ModelRegistry(
            model_class_hf=MistralForCausalLM, model_class_neuron=MistralForSampling, 
        )

except ImportError:
    NEURON_AVAILABLE = False


@register_model("hf-neuron")
class NEURON_HF(LM):
    """
    Enables usage with LLama and Opt on AWS Neuron
    using the HuggingFace Transformers + Transformers neuronx library.
    Tested with neuron 2.15.9
    """

    TORCH_MODEL_CLASS = None
    NEURON_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Optional[str] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision: Optional[str] = "main",
        tp_degree: Optional[int] = None,
        subfolder: Optional[str] = None,
        tokenizer: Optional[str] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[int] = 1,
        low_cpu_mem_usage: Optional[bool] = True,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        cache_dir: Optional[Union[str, os.PathLike]] = HF_HUB_CACHE,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
    ) -> None:
        if not NEURON_AVAILABLE:
            raise Exception(
                "Tried to load neuron model, but neuron is not installed ",
                "please install neuron via pip install transformers-neuron",
                "also make sure you are running on an AWS inf2 instance",
            )
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        self.batch_size_per_gpu = int(batch_size)
        batch_size = int(batch_size)
        if tp_degree is None:
            # execute `neuron-ls --json-output | jq '.[0].nc_count'``
            # to get the number of neuron cores on your instance
            tp_degree = shared_utils.get_nc_count()
        else:
            tp_degree = int(tp_degree)

        assert isinstance(tp_degree, int), (
            f"model_args must include tp_degree. tp_degree must be set to an integer,"
            f" but is tp_degree=`{tp_degree}` type {type(tp_degree)}."
            "Set it to number of neuron cores on your instance."
            " For inf2.xlarge and inf2.8xlarge, set it to `2`."
            " For inf2.24xlarge, set it to `12`."
            " For inf2.48xlarge, set it to `24`."
        )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        arch = getattr(self._config, "architectures")[0]
        try:
            model_class = ModelTypeToClasses[arch].value

            self.TORCH_MODEL_CLASS = model_class.model_class_hf
            self.NEURON_MODEL_CLASS = model_class.model_class_neuron
            self.sampling_fn = model_class.sampling_fn
        except Exception:
            raise Exception(
                "Unsupported model type: ", getattr(self._config, "model_type")
            )

        torch_dtype = utils.get_dtype(dtype)

        assert torch_dtype in [
            torch.float16,
            torch.bfloat16,
        ], "Only float16 and bfloat16 are supported"

        torch_model = self.TORCH_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        )
        self.model_config = copy.deepcopy(torch_model.config)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )

        # Neuron specific code
        if torch_dtype == torch.float16:
            self.amp_dtype = "f16"
        elif torch_dtype == torch.bfloat16:
            self.amp_dtype = "bf16"
        else:
            raise NotImplementedError("Only float16 and bfloat16 are implemented.")
        path_neuron = (
            Path(cache_dir)
            / "neuron-lm-eval"
            / (pretrained.replace("/", "--") + f"-neuron-{self.amp_dtype}-split")
        )
        path_neuron.mkdir(parents=True, exist_ok=True)
        print(f"Splitting neuron model to {path_neuron}")
        save_pretrained_split(torch_model, path_neuron)
        self.neuron_model = LlamaForSampling.from_pretrained(
            path_neuron, batch_size=batch_size, tp_degree=tp_degree, amp=self.amp_dtype
        )
        print(
            f"{'='*20} \n compiling model to neuron with"
            f" tp_degree={tp_degree}, amp={self.amp_dtype} batch_size={batch_size}..."
        )
        self.neuron_model.to_neuron()
        print(f"SUCCESS: neuron model compiled. \n {'='*20}")

        self.truncation = truncation

        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}

        # multigpu data-parallel support when launched with accelerate

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model
        return self.neuron_model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        """device are neuron cores, but the created tensors are on CPU."""
        return "cpu"

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        """ """
        if add_special_tokens is None:
            add_special_tokens = False

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ):
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = False

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)


    def _model_call_v2(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        missing_batch_size: int = self.batch_size - batch_size
        if missing_batch_size:
            # handle the event of input_ids.shape[0] != batch_size
            # Neuron cores expect constant batch_size
            input_ids = torch.concat(
                (
                    input_ids,
                    # add missing_batch_size dummy
                    input_ids[0].repeat(missing_batch_size, 1),
                ),
                dim=0,
            )

        with torch.inference_mode():
            cache_ids = torch.arange(0, sequence_length, dtype=torch.int32).split(1)
            input_ids_split = input_ids.split(1, dim=1)

            return_tensor = torch.stack(
                [
                    self.model(input_ids=input_id, cache_ids=cache_id)
                    for input_id, cache_id in zip(input_ids_split, cache_ids)
                ],
                dim=1,
            )

            if not missing_batch_size:
                return return_tensor
            else:
                return return_tensor[:-missing_batch_size]

    @shared_utils.wrap_constant_batch_size
    def _model_call(self, input_ids: torch.Tensor):
        """
        get logits for the entire sequence

        :param inps: torch.Tensor
            A torch tensor of shape [batch, sequence_cont]
            the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        _, sequence_length = input_ids.shape

        with torch.inference_mode():
            cache_ids = torch.arange(0, sequence_length, dtype=torch.int32).split(1)
            input_ids_split = input_ids.split(1, dim=1)

            return torch.stack(
                [
                    self.model(input_ids=input_id, cache_ids=cache_id)
                    for input_id, cache_id in zip(input_ids_split, cache_ids)
                ],
                dim=1,
            )


    def _model_generate(
        self, context, attention_mask, max_length, stop, **generation_kwargs
    ):
        # we require users to pass do_sample=True explicitly
        # for non-greedy gen. This should be reevaluated when considering beam search.
        # if "do_sample" not in generation_kwargs.keys():
        #     generation_kwargs["do_sample"] = False
        if "top_k" not in generation_kwargs.keys():
            generation_kwargs["top_k"] = None
        # we require users to pass do_sample=True explicitly
        # for non-greedy gen. This should be reevaluated when considering beam search.
        # build stopping criteria
        start_ids = attention_mask.argmax(dim=1)

        stopping_criteria_list = stop_sequences_criteria(
            self.tokenizer, stop, 1, context.shape[0]
        )
        next_token_scores = self.model(context, None, start_ids)
        return shared_utils.sample_loop_llama(
            model=self.model,
            input_ids=context,
            start_ids=start_ids,
            sequence_length=max_length,
            stopping_criteria_list=stopping_criteria_list,
            next_token_scores=next_token_scores,
            eos_token_id=self.eot_token_id,
            top_k=generation_kwargs["top_k"],
            temperature=max(generation_kwargs.get("temperature", 1.0), 1e-6),
            # todo(michael): add support for other kwargs, that are currently not supported
            # e.g. beam_search.
            # **generation_kwargs,
        )

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)
        context_enc = self.tok_encode(context, add_special_tokens=False)

        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context, add_special_tokens=False)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
                    continuation
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []

        adaptive_batch_size = None

        for (string,) in tqdm([req.args for req in requests], disable=(self.rank != 0)):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False, override_bs=None
    ):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        n_reordered_requests = len(re_ord.get_reordered())  # noqa
        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request

        chunks = utils.chunks(
            re_ord.get_reordered(),
            n=self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0,
            fn=None,
        )

        for chunk in tqdm(chunks, disable=(disable_tqdm or (self.rank != 0))):
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []  # noqa
            encoder_attns = []  # noqa

            padding_len_inp = None
            padding_len_cont = None  # noqa
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            batched_inps = utils.pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long, device=self.device
                ).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

                self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def generate_until(self, requests):
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))

        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = utils.chunks(re_ord.get_reordered(), n=self.batch_size)
            for chunk in tqdm(chunks, disable=self.rank != 0):
                contexts, all_gen_kwargs = zip(*chunk)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                    )
                if not until:
                    until = [self.tok_decode(self.eot_token_id)]
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks
                # first stop sequence is used to halt generation upon encountering
                primary_until = [until[0]]

                max_ctx_len = self.max_length - max_gen_toks

                # encode, pad, and truncate contexts for this batch
                context_enc, attn_masks = self.tok_batch_encode(
                    contexts,
                    left_truncate_len=max_ctx_len,
                    truncation=self.truncation,
                )
                context_enc = context_enc.to(self.device)
                attn_masks = attn_masks.to(self.device)

                if "max_length" not in kwargs:
                    kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

                # perform batched generation
                cont = self._model_generate(
                    context=context_enc,
                    attention_mask=attn_masks,
                    stop=primary_until,
                    **kwargs,
                )

                cont_toks_list = cont.tolist()
                for cont_toks, context in zip(cont_toks_list, contexts):
                    # discard context + left-padding toks if using causal decoder-only LM
                    cont_toks = cont_toks[context_enc.shape[1] :]

                    s = self.tok_decode(cont_toks)

                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore '' separator,
                            # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                            s = s.split(term)[0]

                    res[key].append(s)

                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), s
                    )
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)
