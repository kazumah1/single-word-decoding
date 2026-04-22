"""Grid over different configurations.
"""

import os

from sentence_decoding.main import Experiment as Exp

from neuraltrain.utils import update_config

from .defaults import default_config

# logging.getLogger("neuralset").setLevel(logging.DEBUG)

default_params = {
    "infra.cluster": None,
    "use_wandb": False,
    "data.n_timelines": 1,
    "save_checkpoints": False,
    "data.dataset": "Gwilliams2022",
    "transformer_config": {
        "name": "LlamaTransformer",
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
        "num_layers": 12,          # use only 4 of 32 layers to fit in memory
        "freeze_pretrained": True,
        "torch_dtype": "bfloat16",
    },
}

    # "brain_model_config.time_agg_out": "eegnet",
    # "data.feature.model_name": "t5-large",
    # "loss": {"name": "MSELoss", "kwargs": {"reduction": "mean"}},
    # "data.num_workers": 1,
    # "reload_checkpoint": "/storage/users/sdascoli/results/sentence_decoding/paper/data.dataset=Armeni2022,usetransformer=True-e99bc806/best.ckpt",
    # "data.feature": {
    #     "name": "FastTextEmbedding",
    #     "model_folder": "/data/home/sdascoli/word2vec",
    #     "aggregation": "trigger",
    #     "infra": {"folder": default_config["infra"]["folder"]},
    # },
    # "transformer_config.heads": 6,
    #}
default = update_config(default_config, default_params)

params = {
    "data.dataset": "Gwilliams2022",
    "pretrain_mode": "simclr",
    "trainer_config.monitor": "val_pretrain_cnn_loss",
}

if __name__ == "__main__":
    config = update_config(default, params)
    job_name = "|".join([f"{k}={v}" for k, v in params.items()])
    folder = os.path.join(config["infra"]["folder"], "test", job_name)
    config["infra"]["folder"] = folder
    if os.path.exists(folder):
        import shutil

        shutil.rmtree(folder, ignore_errors=True)
    task = Exp(
        **config,
    )
    task.infra.clear_job()
    out = task.run()
