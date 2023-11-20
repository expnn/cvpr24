import torch
from torch import Tensor
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss


__all__ = ['DiceLoss']


class MultiLabelCrossEntropyLoss(_Loss):
    # noinspection PyUnresolvedReferences
    r"""Creates a criterion for multi-label classification.

        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(*, C)`, where :math:`*` means any number of dimensions,
              and C denotes the number of classes.
            - Target: :math:`(*, C)`, same shape as the input.
              One-hot representation of the classification labels.
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(*)`, same shape as the input except for the last dimension.

        Examples::

            >>> loss = MultiLabelCrossEntropyLoss()
            >>> preds = torch.randn(3, 5, requires_grad=True)
            >>> target = torch.gt(torch.randn(3, 5), 0)
            >>> output = loss(preds, target.float())
            >>> output.backward()
        """
    # noinspection PyMethodMayBeStatic
    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = self.multilabel_categorical_crossentropy(preds, target.float())
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

    @staticmethod
    def multilabel_categorical_crossentropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """多标签分类的交叉熵
        说明：y_true 和 y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。如有疑问，请仔细阅读
             https://spaces.ac.cn/archives/7359。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss
