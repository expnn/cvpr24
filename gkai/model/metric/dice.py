from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import AverageMethod


class BinaryDice(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
            self,
            num_classes: Optional[int] = None,
            average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
            zero_division: float = 0.0,
            # include_background: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_average = ("micro", "macro", "weighted", "samples", "none", None)
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
            raise ValueError(f"When you set `average` as '{average}', you have to provide the number of classes.")

        _reduce_options = (AverageMethod.WEIGHTED, None, "none")
        self.reduce = AverageMethod.MACRO if average in _reduce_options else average
        # self.include_background = include_background

        default = lambda: []  # 必须使用 lambda 每次创建一个新的变量, 避免 tp, fp 等状态变量引用同一个变量.
        reduce_fn: Optional[str] = "cat"
        if average != "samples":
            if average == "micro":
                zeros_shape = []
            elif average in ["macro", "weighted", "none", None]:
                zeros_shape = [num_classes]
            else:
                raise ValueError(f'Wrong reduce="{average}"')
            default = lambda: torch.zeros(zeros_shape, dtype=torch.long)
            reduce_fn = "sum"

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default(), dist_reduce_fx=reduce_fn)

        self.average = average
        self.zero_division = zero_division

    def update(self, preds: Tensor, target: Tensor):
        """Update state with predictions and targets."""
        preds = torch.flatten(preds, start_dim=2)  # (B, C, D)
        target = torch.flatten(target, start_dim=2)  # (B, C, D)

        neg_target = torch.logical_not(target)
        neg_preds = torch.logical_not(preds)
        tp = torch.logical_and(preds, target)
        tn = torch.logical_and(neg_preds, neg_target)
        fp = torch.logical_and(preds, neg_target)
        fn = torch.logical_and(neg_preds, target)

        # Update states
        if self.reduce == AverageMethod.SAMPLES:
            self.tp.append(torch.sum(tp, dim=(1, 2)))  # (B, )
            self.fp.append(torch.sum(fp, dim=(1, 2)))
            self.tn.append(torch.sum(tn, dim=(1, 2)))
            self.fn.append(torch.sum(fn, dim=(1, 2)))
        else:
            self.tp += torch.sum(tp, dim=(0, 2))  # (C, )
            self.fp += torch.sum(fp, dim=(0, 2))
            self.tn += torch.sum(tn, dim=(0, 2))
            self.fn += torch.sum(fn, dim=(0, 2))

        return self.tp, self.fp, self.tn, self.fn

    def _get_final_stats(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Performs concatenation on the stat scores if neccesary, before passing them to a compute function."""
        tp = torch.cat(self.tp) if isinstance(self.tp, list) else self.tp
        fp = torch.cat(self.fp) if isinstance(self.fp, list) else self.fp
        tn = torch.cat(self.tn) if isinstance(self.tn, list) else self.tn
        fn = torch.cat(self.fn) if isinstance(self.fn, list) else self.fn
        return tp, fp, tn, fn

    def _compute_dice_elementwise(self, tp, fp, fn):
        dividend = 2.0 * tp + fn + fp
        mask = torch.eq(dividend, 0.0)
        dividend = torch.where(mask, 1e-10, dividend)
        return torch.where(
            mask,
            self.zero_division,
            2.0 * tp / dividend,
        )

    def compute(self) -> Tensor:
        """Computes metric."""
        tp, fp, _, fn = self._get_final_stats()
        if self.average == AverageMethod.MICRO:
            tp = torch.sum(tp)
            fp = torch.sum(fp)
            fn = torch.sum(fn)
            return self._compute_dice_elementwise(tp, fp, fn)

        dice = self._compute_dice_elementwise(tp, fp, fn)
        if self.average in (None, "none"):
            return dice

        if self.average in (AverageMethod.SAMPLES, AverageMethod.MACRO):
            return dice.mean()
        else:
            support = tp + fn
            support_total = support.sum()
            if support_total.item() == 0:
                return dice.mean()
            else:
                return (dice * support) / support_total


if __name__ == '__main__':
    def demo():
        preds = torch.randn((10, 3, 40)) > 0.5
        targets = torch.randn((10, 3, 40)) > 0.5
        metric = BinaryDice(num_classes=3, average=None)
        print(metric.average)
        print(metric.reduce)
        # noinspection PyTypeChecker
        metric.update(preds, targets)
        print(metric.compute())


    demo()
