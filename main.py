import argparse
import yaml
import os
from datetime import datetime
from IterativeCFM import IterativeCFM
from IterativeCFM_with_cuts import IterativeCFM_with_Cuts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    parser.add_argument('path')
    args = parser.parse_args()

    if args.type == "train":
        with open(args.path, 'r') as f:
            parameters = yaml.safe_load(f)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + parameters["run_name"]
        run_dir = os.path.join(dir_path, "results", run_name)

        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
            yaml.dump(parameters, f)

        if parameters.get("efficiency_classifier", False):
            Unfolder = IterativeCFM_with_Cuts(parameters)
        else:
            Unfolder = IterativeCFM(parameters)
        Unfolder.run_dir = run_dir
        Unfolder.run()

    # Todo: Implement reloading and replotting trained model
    elif args.type == "plot":
        raise NotImplementedError

    else:
        raise ValueError

if __name__ == '__main__':
    main()
