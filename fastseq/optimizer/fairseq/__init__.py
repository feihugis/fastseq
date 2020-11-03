# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Automatically apply the optimizations if the supported versions of FairSeq
are detected.
"""
import logging
import sys

from packaging import version

from fastseq.config import FASTSEQ_VERSION, MAX_FAIRSEQ_VERSION, MIN_FAIRSEQ_VERSION
from fastseq.logging import get_logger
from fastseq.utils.api_decorator import OPTIMIZED_CLASSES

logger = get_logger(__name__, logging.INFO)

LATEST_VERSION = 'latest'

def is_supported_fairseq():
    """Check if the installed fairseq is supported.

    Returns:
        a bool value: True indicates the installed fairseq is supported.
    """

    v = version.parse(fairseq.__version__)
    return version.parse(
        MIN_FAIRSEQ_VERSION) <= v <= version.parse(MAX_FAIRSEQ_VERSION)

def apply_fairseq_optimization():
    """Automaticall apply the optimization to the installed fairseq.

    The optimized classes and functions are replaced in runtime.
    """

    if not is_supported_fairseq():
        logger.warning(
            f"fairseq(v{fairseq.__version__}) is not supported by fastseq(v"
            f"{FASTSEQ_VERSION}) yet, please change fairseq to "
            f"v{MIN_FAIRSEQ_VERSION} ~ v{MAX_FAIRSEQ_VERSION}, or check other "
            "versions of fastseq.")
        return

    import fastseq.optimizer.fairseq.beam_search_optimizer  # pylint: disable=import-outside-toplevel
    import fastseq.optimizer.fairseq.generate  # pylint: disable=import-outside-toplevel
    _update_fairseq_model_registration()
    logger.info(f"fairseq(v{fairseq.__version__}) has been optimized by "
                f"fastseq(v{FASTSEQ_VERSION}).")


def _update_fairseq_model_registration():
    """Use the optimized classes to update the registered fairseq models and
    arches.
    """
    for model_name, model_class in MODEL_REGISTRY.items():
        if model_class in OPTIMIZED_CLASSES:
            MODEL_REGISTRY[model_name] = OPTIMIZED_CLASSES[model_class]
            logger.debug(
                "Update the register model {} from {} to {}".format(
                    model_name, model_class, OPTIMIZED_CLASSES[model_class]))

    for arch_name, model_class in ARCH_MODEL_REGISTRY.items():
        if model_class in OPTIMIZED_CLASSES:
            ARCH_MODEL_REGISTRY[arch_name] = OPTIMIZED_CLASSES[model_class]
            logger.debug(
                "Update the register model arch {} from {} to {}".format(
                    arch_name, model_class, OPTIMIZED_CLASSES[model_class]))


try:
    import fairseq # pylint: disable=ungrouped-imports
    from fairseq.models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY # pylint: disable=ungrouped-imports
    from fairseq.sequence_generator import SequenceGenerator # pylint: disable=ungrouped-imports
    apply_fairseq_optimization()
except ImportError as error:
    logger.warning('fairseq can not be imported. Please ignore this warning if '
                   'you are not using fairseq')
except:
    logger.error("Unexpected error: {}".format(sys.exc_info()[0]))
    raise
