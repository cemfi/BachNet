from glob import glob
import os

import data
import inference
import training


def test_BachNet():
    data_loaders = data.get_data_loaders(
        batch_size=training.std_config.batch_size,
        num_workers=training.std_config.num_workers,
        time_grid=training.std_config.time_grid,
        context_radius=training.std_config.context_radius,
        split=training.std_config.split,
        debug=True,
        overwrite=True
    )

    training.std_config.num_epochs = 1
    training.train(training.std_config, data_loaders)

    cp_dirname = sorted(glob('checkpoints/*/'))[-1]
    last_subdir = os.path.basename(os.path.normpath(cp_dirname))
    cp_path = cp_dirname + last_subdir + '_epoch=0001.pt'
    soprano_path = 'data/musicxml/001_soprano.xml'
    score = inference.compose_score(cp_path, soprano_path)
    # score.show()
