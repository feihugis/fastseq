# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on Huggingface to make sure the changes do not affect the
model accuracy.
"""

import logging
import time

import torch
from absl.testing import absltest, parameterized

import fastseq
from fastseq.logging.logging_utils import get_logger
from fastseq.utils.test_utils import TestCaseBase
from fastseq_cli.transformers_utils import calculate_rouge
from transformers import (BartForConditionalGeneration, BartTokenizer)


logger = get_logger(__name__, logging.DEBUG)

class TransformersDynamicBeamSearchTest(TestCaseBase):
    """Test the optimizations on HuggingFace-transformers.
    """
    def setUp(self):
        """Load model, tokenizer and expected output."""

        self.tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn')

        self.source_path = 'data/fastseq_azure/tasks/cnn_dm/raw/test.source'

        self.target_path = 'data/fastseq_azure/tasks/cnn_dm/raw/test.target'
        self.targets = []
        with open(self.target_path, 'rt', encoding="utf-8") as target_file:
            for line in target_file:
                self.targets.append(line.strip())
        self.batch_count = 0

    def _generate(self,
                  slines,
                  max_token_length,
                  num_beams,
                  min_gen_length,
                  max_gen_length,
                  no_repeat_ngram_size,
                  early_stopping):
        """Generate the summaries.

        Args:
            slines (List(str)): a list of input sentences.
            max_token_length (int, optional): max tokenized sentence length.
                                              Defaults to 1024.
            num_beams (int, optional): beam number. Defaults to 4.
            min_gen_length (int, optional): min generation length. Defaults to
                                            55.
            max_gen_length (int, optional): maxium length for the generation
                                            output. Defaults to 199.
            no_repeat_ngram_size (int, optional): size of no repeat gram.
            early_stopping (bool, optional): indicate if the beam search will be
                                             early stopped.

        Returns:
            List(str): a list of generated summaries.
        """
        logger.info("Start to process batch-{}".format(self.batch_count))
        start = time.time()
        with torch.no_grad():
            inputs = self.tokenizer(slines,
                                    max_length=max_token_length,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')

            # Generate Summary
            summary_ids = self.bart_model.generate(
                inputs['input_ids'].cuda(),
                num_beams=num_beams,
                min_length=min_gen_length,
                max_length=max_gen_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping)
            outputs = [self.tokenizer.decode(g) for g in summary_ids]
            self.batch_count += 1
        end = time.time()
        logger.info("Process {} samples in {:.2f} seconds".format(
            len(slines), end - start))
        return outputs

    @parameterized.named_parameters({
        'testcase_name': 'FP32',
        'batch_size': 32,
        'max_token_length': 1024,
        'num_beams_cands': [4, 100],
        'min_gen_length': 55,
        'max_gen_length': 140,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
    })
    def test_beam_search_optimizer(self,
                                   batch_size,
                                   max_token_length,
                                   num_beams_cands,
                                   min_gen_length,
                                   max_gen_length,
                                   no_repeat_ngram_size,
                                   early_stopping):
        """Make sure the changes do not affect the model accuracy.

        Args:
            batch_size (int, optional): batch size. Defaults to 16.
            max_token_length (int, optional): max tokenized sentence length.
                                              Defaults to 1024.
            num_beams (int, optional): beam number. Defaults to 4.
            min_gen_length (int, optional): min generation length. Defaults to
                                            55.
            max_gen_length (int, optional): maxium length for the generation
                                            output. Defaults to 199.
            no_repeat_ngram_size (int, optional): size of no repeat gram.
            early_stopping (bool, optional): indicate if the beam search will be
                                             early stopped.
        """
        self.bart_model.cuda()
        self.bart_model.eval()
        inputs = []
        outputs = []
        slines = []
        outputs = {num_beams: [] for num_beams in num_beams_cands}
        timing = {num_beams: 0 for num_beams in num_beams_cands}

        fout = open("debug_dbs_outputs_{}.log".format(time.time()), mode='w')

        with open(self.source_path, 'rt', encoding="utf-8") as source:
            for sline in source:
                inputs.append(sline)
                slines.append(sline)
                if (len(slines) % batch_size > 0 and
                    len(inputs) != len(self.targets)):
                    continue

                for num_beams in num_beams_cands:
                    start = time.time()
                    outputs[num_beams].extend(self._generate(
                        slines,
                        max_token_length,
                        num_beams,
                        min_gen_length,
                        max_gen_length,
                        no_repeat_ngram_size,
                        early_stopping))
                    end = time.time()
                    timing[num_beams] = timing[num_beams] + end - start

                for i in range(len(inputs) - len(slines), len(inputs), 1):
                    debug_str = ("\n-----------------------------------------\n"
                    "\n--SOURCE \n \n {} \n"
                    "\n--TARGET \n \n {} \n".format(inputs[i], self.targets[i]))

                    for num_beams in num_beams_cands:
                        output = outputs[num_beams][i].replace('<pad>', '')
                        rouge = calculate_rouge([output], [self.targets[i]])
                        debug_str += "\n--BEAM={} ROUGE={} \n \n G-{}-{} \n".format(
                            num_beams, rouge, num_beams, output)

                    fout.write(debug_str + '\n')
                    fout.flush()
                    logger.debug(debug_str)

                slines = []

            for num_beams in num_beams_cands:
                logger.debug(
                "NUM_BEAMS={}: process {} samples with speed {:.2f} samples/second"
                .format(num_beams, len(inputs), len(inputs) / timing[num_beams]))

        fout.close()


if __name__ == "__main__":
    absltest.main()
