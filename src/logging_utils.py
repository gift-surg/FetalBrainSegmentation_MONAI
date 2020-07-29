import torch

from monai.utils.misc import is_scalar
import warnings

DEFAULT_KEY_VAL_FORMAT = '{}: {:.4f} '
DEFAULT_TAG = 'Loss'


def my_iteration_print_logger(engine):
    """Execute iteration log operation based on Ignite engine.state data.
    Print the values from ignite state.logs dict.
    Default behavior is to print loss from output[1], skip if output[1] is not loss.

    Args:
        engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

    """
    key_var_format = DEFAULT_KEY_VAL_FORMAT
    tag_name = DEFAULT_TAG
    loss = engine.state.output
    if loss is None:
        return  # no printing if the output is empty

    out_str = ''
    if isinstance(loss, dict):  # print dictionary items
        for name in sorted(loss):
            value = loss[name]
            if not is_scalar(value):
                warnings.warn('ignoring non-scalar output in StatsHandler,'
                              ' make sure `output_transform(engine.state.output)` returns'
                              ' a scalar or dictionary of key and scalar pairs to avoid this warning.'
                              ' {}:{}'.format(name, type(value)))
                continue  # not printing multi dimensional output
            out_str += key_var_format.format(name, value.item() if torch.is_tensor(value) else value)
    else:
        if is_scalar(loss):  # not printing multi dimensional output
            out_str += key_var_format.format(tag_name, loss.item() if torch.is_tensor(loss) else loss)
        else:
            warnings.warn('ignoring non-scalar output in StatsHandler,'
                          ' make sure `output_transform(engine.state.output)` returns'
                          ' a scalar or a dictionary of key and scalar pairs to avoid this warning.'
                          ' {}'.format(type(loss)))

    if not out_str:
        return  # no value to print

    num_iterations = engine.state.epoch_length
    # current_iteration = (engine.state.iteration - 1) % num_iterations + 1
    current_iteration = engine.state.iteration
    current_epoch = engine.state.epoch
    num_epochs = engine.state.max_epochs

    base_str = "Epoch: {}/{}, Iter: {} --".format(
        current_epoch,
        num_epochs,
        current_iteration,
        num_iterations)

    engine.logger.info(' '.join([base_str, out_str]))