#!/bin/bash
# file: docker-nv-example.sh
 
#sudo docker pull nvcr.io/nvidia

#docker run --gpus all -it --rm -v ~/minor_:docker-nv.sh nvcr.io/nvidia/tensorflow:<21.01>
 
mkdir -p $HOME/digits_workdir/jobs
 
cat <<EOF > $HOME/digits_workdir/digits_config_env.sh
DIGITS Configuration File
DIGITS_JOB_DIR=$HOME/digits_workdir/jobs
DIGITS_LOGFILE_FILENAME=$HOME/digits_workdir/digits.log
EOF
 
docker run --gpus all --rm -ti --name=${USER}_digits -p 5000:5000 \
  -u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME \
  --env-file=${HOME}/digits_workdir/digits_config_env.sh \
  -v /datasets:/digits_data:ro \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/digits:17.05
