import torch

from ignite.engine.engine import Engine
from ignite.engine import _prepare_batch

from sliding_window_inference import sliding_window_inference


def create_supervised_trainer_with_clipping(model, optimizer, loss_fn,
                                            device=None, non_blocking=False,
                                            prepare_batch=_prepare_batch,
                                            output_transform=lambda x, y, y_pred, loss: loss.item(),
                                            clip_norm=None):
    """
    Factory function for creating a trainer for supervised models with possibility to define gradient clipping.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`
        clip_norm (float, optional): value to use to clip the norm of the gradients. If None, no clipping is applied

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if clip_norm is not None:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            except:
                print("Gradient clipping failed.")
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def _sliding_window_inference(x, model):
    """
    Prepare inputs for sliding window inference
    :param x: input data
    :param model: predictor
    :return: inference for input x using the provided model and a sliding window approach
    """
    return sliding_window_inference(inputs=x, roi_size=(96, 96, 1), sw_batch_size=1, predictor=model)


def create_evaluator_with_sliding_window(model, metrics=None,
                                         device=None, non_blocking=False,
                                         prepare_batch=_prepare_batch,
                                         sliding_window_inference=_sliding_window_inference,
                                         output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models, using a sliding window inference approach.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        sliding_window_inference (callable, optional): function that receives the input x and the model and returns the
            inference result for x using a sliding window approach defined by the function.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = sliding_window_inference(x, model)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
