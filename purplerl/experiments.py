from typing import Dict
import pandas as pd
from simplerl.simple import *

def get_exp_name(args: Dict[str, str], df: pd.DataFrame) -> str:
    def is_unique(s: pd.Series):
        it = iter(s)
        value = next(it)
        for v in it:
            if v != value:
                return False
        return True

    result = ""
    for key, value in args.items():
        if not is_unique(df[key]):
            result += f"{key}_{value}-"
    result = result[:-1]
    return result

def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=str, default=None)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    experiments_df = pd.read_json(args.experiments)

    for run in range(5):
        seed = run*500

        for args in experiments_df.to_dict("records"):

            env_name = args.pop("env_name")

            exp_name = get_exp_name(args, experiments_df)

            logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=f"./{env_name}")
            trainer = Trainer(logger_kwargs)

            trainer.train(seed=seed, **args)


if __name__ == '__main__':
    run()
