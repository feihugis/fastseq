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
from fairseq.models.bart.model import BARTModel

import fastseq
from fastseq.config import FASTSEQ_LOG_FORMAT
from fastseq.logging.logging_utils import get_logger
from fastseq.utils.test_utils import TestCaseBase, CACHED_BART_MODEL_PATHS
from fastseq.utils.visualize_util import plot_beam_score_trackings
from fastseq_cli.transformers_utils import calculate_rouge
from transformers import (BartForConditionalGeneration, BartTokenizer)
from transformers import GPT2Tokenizer


logger = get_logger(__name__, logging.DEBUG)

class TransformersDynamicBeamSearchTest(TestCaseBase):
    """Test the optimizations on HuggingFace-transformers.
    """
    def setUp(self):
        """Load model, tokenizer and expected output."""

        self.bart = BARTModel.from_pretrained(
            CACHED_BART_MODEL_PATHS['bart.large.cnn'],
            checkpoint_file='model.pt')

        self.tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-large-cnn')
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn')

        self.source_path = 'data/fastseq_azure/tasks/cnn_dm/raw/test.source'
        self.target_path = 'data/fastseq_azure/tasks/cnn_dm/raw/test.target'
        # self.source_path = 'tests/optimizer/fairseq/data/cnndm_128.txt'
        # self.source_path = 'tests/optimizer/transformers/debug.source'
        # self.target_path = 'tests/optimizer/transformers/debug.target'

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
                  early_stopping,
                  length_penalty,
                  use_cache,
                  beam_token_prob_thresh,):
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
            # fairseq_inputs = [self.bart.encode(sline) for sline in slines]
            # assert (fairseq_inputs[0] == inputs['input_ids']).all().item(), "Input tokens are different between transformers and fairseq"

            # Generate Summary
            summary_ids, beam_scores = self.bart_model.generate(
                inputs['input_ids'].cuda(),
                num_beams=num_beams,
                min_length=min_gen_length,
                max_length=max_gen_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
                use_cache=use_cache,
                beam_token_prob_thresh=beam_token_prob_thresh,)
            # outputs = [self.tokenizer.decode(g) for g in summary_ids]
            outputs = [self.bart.decode(g[g != 1]) for g in summary_ids]
            self.batch_count += 1
        end = time.time()
        logger.info("Process {} samples in {:.2f} seconds".format(
            len(slines), end - start))
        return outputs, beam_scores

    @parameterized.named_parameters({
        'testcase_name': 'FP32',
        'batch_size': 16,
        'max_token_length': 1024,
        'num_beams_cands': [8],
        'min_gen_length': 55,
        'max_gen_length': 140,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
        'length_penalty': 0.1,
        'use_cache': True,
        'is_tmp': False,
        'beam_token_prob_thresh': 0.95,
    })
    def test_beam_search_optimizer(self,
                                   batch_size,
                                   max_token_length,
                                   num_beams_cands,
                                   min_gen_length,
                                   max_gen_length,
                                   no_repeat_ngram_size,
                                   early_stopping,
                                   length_penalty,
                                   use_cache,
                                   is_tmp,
                                   beam_token_prob_thresh):
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
            beam_token_prob_thresh (float, optional): threshhold used to define
                                                      the valid range for token
                                                      probability.
        """
        self.bart_model.cuda()
        self.bart_model.eval()
        inputs = []
        slines = []
        outputs = {num_beams: [] for num_beams in num_beams_cands}
        beam_score_trackings = {num_beams: [] for num_beams in num_beams_cands}
        rouges = {num_beams: [] for num_beams in num_beams_cands}

        timing = {num_beams: 0 for num_beams in num_beams_cands}

        cur_time = time.time() if not is_tmp else 999

        base_filename = "{}dbs_bs{}_nb{}_lp{}_maxgl{}_mingl{}_nrn{}_uc{}_es{}_btpt{}".format(
            "tmp/" if is_tmp else "",
            batch_size,
            '+'.join(str(num_beams) for num_beams in num_beams_cands),
            length_penalty,
            max_gen_length,
            min_gen_length,
            no_repeat_ngram_size,
            use_cache,
            early_stopping,
            beam_token_prob_thresh,
        )
        output_file_name = "debug/{}_OUTPUTS_{}.log".format(
            base_filename, cur_time)
        deboug_file_name = "debug/{}_DEBUG_{}.log".format(
            base_filename, cur_time)
        beam_score_trackings_img = "vis/{}_{}.png".format(
            base_filename, cur_time)

        # Update log file handler
        fh = logging.FileHandler(deboug_file_name, mode='w')
        formatter = logging.Formatter(FASTSEQ_LOG_FORMAT)
        fh.setFormatter(formatter)
        root_logger = logging.getLogger()
        for hdlr in root_logger.handlers[:]:
            if isinstance(hdlr,logging.FileHandler):
                root_logger.removeHandler(hdlr)
        root_logger.addHandler(fh)

        fout = open(output_file_name, mode='w')

        test_fout = open("test_transformers_{}_100.log".format(length_penalty), mode='w')
        with open(self.source_path, 'rt', encoding="utf-8") as source:
            for sline in source:
                inputs.append(sline.strip())
                slines.append(sline.strip())
                if (len(slines) % batch_size > 0 and
                    len(inputs) != len(self.targets)):
                    continue

                for num_beams in num_beams_cands:
                    start = time.time()
                    output, beam_score_tracking = self._generate(
                        slines,
                        max_token_length,
                        num_beams,
                        min_gen_length,
                        max_gen_length,
                        no_repeat_ngram_size,
                        early_stopping,
                        length_penalty=length_penalty,
                        use_cache=use_cache,
                        beam_token_prob_thresh=beam_token_prob_thresh)
                    outputs[num_beams].extend([line.strip() for line in output])
                    beam_score_trackings[num_beams].extend(beam_score_tracking)
                    end = time.time()
                    timing[num_beams] = timing[num_beams] + end - start

                for i in range(len(inputs) - len(slines), len(inputs), 1):
                    debug_str = ("\n---------------------------------------\n\n"
                    "--SOURCE\n\n{}\n\n"
                    "--TARGET\n\n{}\n\n".format(inputs[i], self.targets[i]))

                    for num_beams in num_beams_cands:
                        output = outputs[num_beams][i].replace(
                            '<pad>', '').replace('<s>', '').replace('</s>', '')
                        test_fout.write(output + '\n')
                        test_fout.flush()
                        rouge = calculate_rouge([output], [self.targets[i]])
                        rouges[num_beams].append(rouge)
                        debug_str += "--BEAM={} ROUGE={}\n\nH-{}-{}\n\n".format(
                            num_beams, rouge, num_beams, output)

                    logger.debug(debug_str)
                    fout.write(debug_str + '\n')
                    fout.flush()

                slines = []

                if is_tmp and len(outputs[num_beams_cands[0]]) >= 32:
                    break

            for num_beams in num_beams_cands:
                logger.debug(
                "NUM_BEAMS={}: process {} samples with speed {:.2f} samples/second"
                .format(num_beams, len(inputs), len(inputs) / timing[num_beams]))

        fout.close()
        test_fout.close()

        # plot_beam_score_trackings(
        #     beam_score_trackings=beam_score_trackings,
        #     rouges=rouges,
        #     img_path=beam_score_trackings_img)
        logger.debug("Output img file: {}".format(beam_score_trackings_img))
        logger.debug("Output log file: {}".format(deboug_file_name))


if __name__ == "__main__":
    absltest.main()
