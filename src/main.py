import warnings

import hydra
from models.modeplait.model import ModePlait
from omegaconf import DictConfig

from src.models.modeplait.run import run as modeplait_simulator
from src.utils.io_helper import IOHelper
from src.utils.params_set import params_set
from src.utils.preprocessor import preprocessing

warnings.simplefilter("ignore")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to simulate real-time forecasting.

    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    """
    # initialize IOHelper
    ioh = IOHelper(io_cfg=cfg.io)
    ioh.init_dir()

    # prepare for simulation
    #   - load data
    #   - preprocessing
    #   - set params
    #   - others
    data = ioh.load_data(io_cfg=cfg.io)
    B_true = None
    if cfg.io.input_dir == "synthetics":
        data, B_true = data
    data = preprocessing(data=data, prep_cfg=cfg.prep)

    print(f"DATASETS: {cfg.io.input_dir}_{cfg.io.uuid}")
    print(f"PREPROCESSED: {data.shape}")
    print(f"MODEL: {cfg.model.name}")
    cfg = params_set(data, cfg)
    ioh.out_dir = cfg.io.out_dir

    # load model
    model = ModePlait()

    # create output directory
    if cfg.save:
        ioh.mkdir()

    # simulate
    results = modeplait_simulator(data, model, cfg)
    if B_true is not None:
        results["B_true"] = B_true

    # save results
    if cfg.save and results is not None:
        for key, value in results.items():
            ioh.savepkl(obj=value, name=key)


if __name__ == "__main__":
    main()
