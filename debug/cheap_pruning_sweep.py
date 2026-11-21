import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import Runner

CHECKPOINT_SUFFIX = "Aug23_11-28-31_108000"  # best checkpoint from the base run
RATIOS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 0.4 = baseline

def main(gpu_id):
    runner = Runner("train_exp3_base", gpu_id)
    model = runner.initialize_model(CHECKPOINT_SUFFIX)
    examples_train, examples_dev = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    for ratio in RATIOS:
        runner.config["top_span_ratio"] = ratio
        print(f"\n=== top_span_ratio = {ratio} ===")
        # official=False skips the slow external CorefUD scorer -- the mention
        # recall diagnostics print regardless, which is what this sweep is for.
        runner.evaluate(model, examples_dev, stored_info, 0, official=False,
                         conll_path=runner.config["conll_eval_path"])

if __name__ == "__main__":
    main(int(sys.argv[1]))
