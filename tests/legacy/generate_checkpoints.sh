#!/bin/bash
# Usage:
# 1. Generate checkpoints with one or more specified PL versions:
#    bash generate_checkpoints.sh 1.0.2 1.0.3 1.0.4
# 2. Generate checkpoints with the PL version installed in your environment:
#    bash generate_checkpoints.sh
set -e

LEGACY_PATH=$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
ENV_PATH=$LEGACY_PATH/vEnv
echo LEGACY_PATH: $LEGACY_PATH
echo ENV_PATH: $ENV_PATH

function create_and_save_checkpoint {
  python --version
  python -m pip --version
  python -m pip list

  python $LEGACY_PATH/simple_classif_training.py

  cp $LEGACY_PATH/simple_classif_training.py $LEGACY_PATH/checkpoints/$pl_ver
  mv $LEGACY_PATH/checkpoints/$pl_ver/lightning_logs/version_0/checkpoints/*.ckpt $LEGACY_PATH/checkpoints/$pl_ver/
  rm -rf $LEGACY_PATH/checkpoints/$pl_ver/lightning_logs
}

# iterate over all arguments assuming that each argument is version
for pl_ver in "$@"
do
  echo processing version: $pl_ver

  # Don't install/update anything before activating venv
  # to avoid breaking any existing environment.
  python -m venv $ENV_PATH
  source $ENV_PATH/bin/activate

  python -m pip install pytorch_lightning==$pl_ver -r $LEGACY_PATH/requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

  create_and_save_checkpoint

  deactivate
  rm -rf $ENV_PATH
done

# use the PL installed in the environment if no PL version is specified
if [[ -z "$@" ]]; then
  pl_ver=$(python -c "import pytorch_lightning as pl; print(pl.__version__)")
  echo processing version: $pl_ver

  python -m pip install -r $LEGACY_PATH/requirements.txt

  create_and_save_checkpoint
fi
