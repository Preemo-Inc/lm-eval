import numpy as np
import random
from lm_eval.base import rf, mean
from . common import HFTask

"""
NOTE: This evaluation of Winograd Schema Challenge is based on `partial evaluation`
as described by Trinh & Le in Simple Method for Commonsense Reasoning (2018).
See: https://arxiv.org/abs/1806.02847
"""


class WinogradSchemaChallenge273(HFTask):
    DATASET_PATH = "winograd_wsc"
    DATASET_NAME = "wsc273"

    upper_pronouns = ["A", "An", "The", "She", "He",
                      "It", "They", "My", "His", "Her", "Their"]

    def __init__(self):
        super().__init__()
        self.data = self.__clean_data()

    def __clean_data(self):
        # The HF implementation of `wsc273` is not `partial evaluation` friendly.
        data = []
        for doc in self.data["test"]:
            doc["text"] = doc["text"].replace("  ", " ")
            doc["options"][0] = self.__normalize_option(doc, doc["options"][0])
            doc["options"][1] = self.__normalize_option(doc, doc["options"][1])
            data.append(doc)
        return {"test": data}

    def __normalize_option(self, doc, option):
        # Append `'s` to possessive determiner based options.
        if doc["pronoun"].lower() in ["my", "his", "her", "our", "their"]:
            option += "'s"
        # Appropriately lowercase the pronoun in the option.
        pronoun = option.split()[0]
        start_of_sentence = doc["text"][doc['pronoun_loc'] - 2] == '.'
        if not start_of_sentence and pronoun in self.upper_pronouns:
            return option.replace(pronoun, pronoun.lower())
        return option

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: redo description
        return "Winograd schema sentence with correct continuation. True. Winograd schema sentence with incorrect continuation. False."

    def fewshot_examples(self, k):
        # NOTE: `super().fewshot_examples` samples from training docs which are
        # not available for this test-set-only dataset.
        return random.sample(list(self.test_docs()), k)

    def doc_to_text(self, doc):
        return self.partial_context(doc, doc["options"][doc["label"]])

    @classmethod
    def partial_context(cls, doc, option):
        # Substitute the pronoun in the original text with the specified
        # option and ignore everything after.
        return doc["text"][:doc["pronoun_loc"]] + option

    def doc_to_target(self, doc):
        return self.partial_target(doc)

    @classmethod
    def partial_target(cls, doc):
        # The target is everything after the document specified pronoun.
        start_index = doc["pronoun_loc"] + len(doc["pronoun"])
        return " " + doc["text"][start_index:].strip()

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        target = self.partial_target(doc)
        lls = []
        for option in doc["options"]:
            partial_ctx = self.partial_context(doc, option)
            full_ctx = self.append_context(ctx, partial_ctx)
            lls.append(rf.loglikelihood(full_ctx, target)[0])
        return lls

    @classmethod
    def append_context(cls, ctx, partial_ctx):
        ctx = ctx.split("\n\n")  # Each fewshot context is on its own new line.
        ctx.pop()  # Remove the correct context put in by `doc_to_text`.
        return "\n\n".join([*ctx, partial_ctx]) if ctx else partial_ctx

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        return {
            "acc": np.argmax(results) == doc["label"]
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "acc": True
        }
