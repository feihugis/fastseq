# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on FairSeq to make sure the changes do not affect the
model accuracy.
"""

import logging
import os
import time

import torch
from absl.testing import absltest, parameterized
from fairseq.models.bart.model import BARTModel

import fastseq
from fastseq.logging import get_logger
from fastseq.utils.file_utils import decompress_file, make_dirs, wget
from fastseq.utils.test_utils import (BART_MODEL_URLS, CACHED_BART_MODEL_DIR,
                                      CACHED_BART_MODEL_PATHS, TestCaseBase)

logger = get_logger(__name__, logging.DEBUG)

class FairseqBeamSearchOptimizerTest(TestCaseBase):
    """Test the optimizations on FairSeq

    `bart.large.cnn` model is used for benchmarking. If it does not exist, it
    will be downloaded first. As the the model is big, it will take a while to
    download. Once downloaded, it will be cached for future usage.
    """

    def setUp(self):
        """set up the test environment"""

        super(FairseqBeamSearchOptimizerTest, self).setUp()
        # TODO: create a dummy model instead of loading a large-size model.
        if not os.path.exists(CACHED_BART_MODEL_PATHS['bart.large.cnn']):
            make_dirs(CACHED_BART_MODEL_DIR, exist_ok=True)
            tar_model_path = os.path.join(CACHED_BART_MODEL_DIR,
                                          'bart.large.cnn.tar.gz')
            with open(tar_model_path, 'xb') as tar_model_file:
                wget(BART_MODEL_URLS['bart.large.cnn'], tar_model_file)
            decompress_file(tar_model_path, CACHED_BART_MODEL_DIR)

        self.bart = BARTModel.from_pretrained(
            CACHED_BART_MODEL_PATHS['bart.large.cnn'],
            checkpoint_file='model.pt')

        self.source_path = 'data/fastseq_azure/tasks/cnn_dm/raw/test.source'

        # read the expected output.
        self.expected_output_path = 'data/fastseq_azure/tasks/cnn_dm/raw/test.target'  # pylint: disable=line-too-long
        self.expected_outputs = []
        with open(self.expected_output_path, 'rt',
                  encoding="utf-8") as expected_output_file:
            for line in expected_output_file:
                self.expected_outputs.append(line.strip())

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'beam_size': 50,
        'batch_size': 16,
        'need_attn': False,
        'lenpen': 0.2,
        'max_len_b': 140,
        'min_len': 55,
        'no_repeat_ngram_size': 3
    })
    def test_beam_search_optimizer(self, beam_size, batch_size, need_attn,
                                   lenpen, max_len_b, min_len,
                                   no_repeat_ngram_size):
        """Make sure the changes do not affect the model accuracy.

        Args:
            beam_size (int): beam size.
            batch_size (int): batch size.
            need_attn (bool): indicate if attention is needed.
            lenpen (float): length penalty, where <1.0 favors shorter, >1.0
                            favors longer sentences.
            max_len_b (int): max length of generated text.
            min_len (int): min length of generated text.
            no_repeat_ngram_size (int): size of no repeat gram.
        """
        self.bart.model.make_generation_fast_(beamable_mm_beam_size=beam_size,
                                              need_attn=need_attn)
        self.bart.cuda()
        self.bart.eval()
        count = 0
        outputs = []
        fout = open(
            "debug_fairseq_outputs_bs{}_nb{}_lp{}_maxgl{}_mingl{}_nrn{}_{}.log".format(
                batch_size,
                beam_size,
                lenpen,
                max_len_b,
                min_len,
                no_repeat_ngram_size,
                time.time()),
            mode='w')

        with open(self.source_path, 'rt', encoding="utf-8") as source:
            slines = []
            torch.cuda.synchronize()
            for sline in source:
                slines.append(sline.strip())
                count += 1
                if count % batch_size != 0:
                    continue
                logger.info("Have processed {} samples".format(count))
                with torch.no_grad():
                    hypotheses_batch = self.bart.sample(
                        slines,
                        beam=beam_size,
                        lenpen=lenpen,
                        max_len_b=max_len_b,
                        min_len=min_len,
                        no_repeat_ngram_size=no_repeat_ngram_size)
                    hypotheses_batch = [
                        output.strip() for output in hypotheses_batch
                    ]
                outputs.extend(hypotheses_batch)
                for hypo in hypotheses_batch:
                    fout.write(hypo + '\n')
                    fout.flush()
                slines = []

                # break

            if len(slines) > 0:
                with torch.no_grad():
                    hypotheses_batch = self.bart.sample(
                        slines,
                        beam=beam_size,
                        lenpen=lenpen,
                        max_len_b=max_len_b,
                        min_len=min_len,
                        no_repeat_ngram_size=no_repeat_ngram_size)
                    hypotheses_batch = [
                        output.strip() for output in hypotheses_batch
                    ]
                outputs.extend(hypotheses_batch)
                for hypo in hypotheses_batch:
                    fout.write(hypo + '\n')
                    fout.flush()

            torch.cuda.synchronize()

        fout.close()


if __name__ == "__main__":
    absltest.main()


# A French prosecutor says he is not aware of any video footage from on board the plane. German daily Bild and Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie spokesman calls the reports "completely wrong" and "unwarranted" German airline Lufthansa says co-pilot Andreas Lubitz battled depression years before he took the controls.
# A French prosecutor says he is not aware of any video footage from on board the plane. German daily Bild and Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie spokesman calls the reports "completely wrong" and "unwarranted" German airline Lufthansa says co-pilot Andreas Lubitz battled depression years before he took the controls.
