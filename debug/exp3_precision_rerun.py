import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import Runner

# (config_name, best checkpoint suffix), recovered from each condition's training
# log ("**********Best model evaluation**********" block). Post-pruning recall from
# these checkpoints was cross-checked against experiment-results/experiment3/
# experiment3-results.md and matches exactly, confirming these are the right ones.
CONDITIONS = [
    ("train_exp3_base", "Aug23_11-28-31_108000"),
    ("train_exp3_width_small", "Aug24_16-01-21_49000"),
    ("train_exp3_width_extended", "Aug25_13-14-25_119000"),
    ("train_exp3_pruning_tight", "Aug27_13-40-02_119000"),
    ("train_exp3_pruning_loose", "Aug28_00-10-24_120000"),
    ("train_exp3_context_small", "Sep03_00-26-48_114000"),
    ("train_exp3_context_extended", "Sep03_14-22-08_122000"),
    ("train_exp3_context_wide", "Sep04_16-47-58_113000"),
]


def main(gpu_id):
    for config_name, suffix in CONDITIONS:
        print(f"\n=== {config_name} (checkpoint {suffix}) ===")
        runner = Runner(config_name, gpu_id)
        model = runner.initialize_model(suffix)
        examples_train, examples_dev = runner.data.get_tensor_examples()
        stored_info = runner.data.get_stored_info()
        # official=False skips the slow external CorefUD scorer -- MUC/B3/CEAFe/CoNLL
        # are already known from the original runs and unaffected by this logging-only
        # change. The new mention precision diagnostics print regardless.
        runner.evaluate(model, examples_dev, stored_info, 0, official=False,
                         conll_path=runner.config["conll_eval_path"])


if __name__ == "__main__":
    main(int(sys.argv[1]))
