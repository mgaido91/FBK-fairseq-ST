# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils, metrics

from . import FairseqCriterion, register_criterion


@register_criterion('knowledge_distillation')
class CrossEntropyKnowledgeDistillationCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        # Lambda ranges between 0.0 and 1.0. 0.0 means that we only use the ground
        # truth labels (ie. it is the same as the normal cross entropy); 1.0 means
        # that only the teacher output is taken in account.
        self._lambda = args.kd_lambda
        self.temperature = args.kd_temperature

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--kd-lambda', default=0., type=float, metavar='D',
                            help='Value for lambda in Knowledge Distillation '
                                 '(ie. the weight of teacher output vs ground truth)')
        parser.add_argument('--kd-temperature', default=1., type=float, metavar='D',
                            help='Temperature to be used. Temperature is used to soften the nets '
                                 'output in order to increase the dark knowledge effect. A temperature '
                                 ' of 1 (default), is equivalent not to use the temperature.')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        target = model.get_targets(sample, net_output).view(-1)

        # KD from the teacher
        if self._lambda > 0.0:
            net_output_scaled = (net_output[0] / self.temperature, net_output[1])
            lprobs = model.get_normalized_probs(net_output_scaled, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            teacher_idxs = sample['teacher_output'][0]
            teacher_outs = sample['teacher_output'][1]
            teacher_probs = F.softmax(teacher_outs / self.temperature, dim=-1)
            teacher_idxs = teacher_idxs.view(-1, teacher_idxs.shape[-1])
            teacher_probs = teacher_probs.view(-1, teacher_probs.shape[-1])

            lprobs_selected = lprobs.gather(dim=-1, index=teacher_idxs.long())
            teacher_loss = - (lprobs_selected * teacher_probs).sum(dim=-1)

            # Ignore paddings
            mask = target != self.padding_idx
            teacher_loss = teacher_loss * mask.type(teacher_loss.dtype)
        else:
            teacher_loss = 0.0

        if self._lambda < 1.0:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            truth_loss = F.nll_loss(
                lprobs, target, size_average=False, ignore_index=self.padding_idx, reduce=False)
        else:
            truth_loss = 0.0

        if isinstance(teacher_loss, torch.Tensor) and isinstance(truth_loss, torch.Tensor):
            assert teacher_loss.shape == truth_loss.shape
        loss = (1.0 - self._lambda) * truth_loss + self._lambda * teacher_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        if reduce:
            loss = loss.sum()
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))
