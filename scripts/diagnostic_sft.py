"""Bounded teacher-loss ablation; keeps inputs and batch construction identical."""
import sys
import sft


def main():
    if "--final-move-only" in sys.argv:
        sys.argv.remove("--final-move-only")
        original = sft.sft_losses

        def final_loss(model, x, targets, move_weight):
            import torch
            positions = torch.arange(targets.shape[1], device=targets.device).expand_as(targets)
            final = positions.masked_fill(targets == -1, -1).max(1).values
            masked = torch.full_like(targets, -1)
            rows = torch.arange(targets.shape[0], device=targets.device)
            masked[rows, final] = targets[rows, final]
            return original(model, x, masked, move_weight)

        sft.sft_losses = final_loss
    sft.main()


if __name__ == "__main__":
    main()
