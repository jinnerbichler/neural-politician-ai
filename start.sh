#!/usr/bin/env bash

INSTANCE_NAME="neural-politician"

set -e    # Exits immediately if a command exits with a non-zero status.

if [ "$1" == "create" ]; then

    # https://cloud.google.com/compute/pricing
    gcloud compute instances create ${INSTANCE_NAME} \
    --machine-type n1-standard-4 --zone us-east1-d \
    --accelerator type=nvidia-tesla-k80,count=2 \
    --boot-disk-size=100GB --image gpu-image \
    --maintenance-policy TERMINATE --restart-on-failure \
    --preemptible

    while [ -n "$(gcloud compute ssh ${INSTANCE_NAME} --command "echo ok" --zone us-east1-d 2>&1 > /dev/null)" ]; do
        echo "Waiting for VM to be available"
        sleep 1.0
    done

elif [ "$1" == "delete" ]; then

    gcloud compute instances delete --zone us-east1-d ${INSTANCE_NAME} --quiet

elif [ "$1" == "upload-data" ]; then

    gcloud compute scp ./data/ ${INSTANCE_NAME}:~/ --recurse --zone us-east1-d
#    gcloud compute ssh ${INSTANCE_NAME} --command="sudo chmod -R 777 /data" --zone us-east1-d

elif [ "$1" == "upload-models" ]; then

#    gcloud compute ssh ${INSTANCE_NAME} --command="sudo chmod 777 ./models" --zone us-east1-d
    gcloud compute scp ./models/ ${INSTANCE_NAME}:~/ --recurse --zone us-east1-d

elif [ "$1" == "init" ]; then

    # https://cloud.google.com/compute/pricing
    gcloud compute instances create ${INSTANCE_NAME} \
    --machine-type n1-standard-4 --zone us-east1-d \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --boot-disk-size=100GB \
    --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud \
    --maintenance-policy TERMINATE --restart-on-failure \
    --preemptible

    while [ -n "$(gcloud compute ssh ${INSTANCE_NAME} --command "echo ok" --zone us-east1-d 2>&1 > /dev/null)" ]; do
        echo "Waiting for VM to be available"
        sleep 1.0
    done

    # Sleep to be sure
    sleep 1.0

    gcloud compute scp ./start.sh ./daemon.json ${INSTANCE_NAME}:~/ --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="~/start.sh init-remote" --zone us-east1-d

elif [ "$1" == "init-remote" ]; then

    echo "Checking for CUDA and installing."
    # Check for CUDA and try to install.
    if ! dpkg-query -W cuda-9-1; then
      curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
      sudo dpkg -i ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
      sudo apt-get update
      sudo apt-get install cuda-9-1 -y --allow-unauthenticated
    fi
    sudo nvidia-smi -pm 0
    sudo nvidia-smi -ac 2505,875
    # On instances with NVIDIA® Tesla® K80 GPU: disable autoboost
    sudo nvidia-smi --auto-boost-default=DISABLED
    nvidia-smi

    # Installing Docker
    echo "Installing Docker and Docker Compose"
    sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get -y install docker-ce=17.12.1~ce-0~ubuntu
    docker --version
    sudo curl -L https://github.com/docker/compose/releases/download/1.19.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    docker-compose --version

    # Installing Nvidia Docker
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo pkill -SIGHUP dockerd

    # https://github.com/NVIDIA/nvidia-docker/issues/262
    sudo nvidia-modprobe -u -c=0

    # Set default runtime
    sudo mv ~/daemon.json /etc/docker/daemon.json
    sudo service docker restart

    # Test nvidia-smi with the latest official CUDA image
    sudo docker run --rm nvidia/cuda nvidia-smi

    # Fetch and cache base image
    sudo docker pull jinnerbichler/neural-politician:latest

elif [ "$1" == "deploy" ]; then

    gcloud compute scp  \
        ./docker-compose.yml \
        ./Dockerfile \
        ./env \
        char_rnn.py \
        word_rnn.py \
        speech_data.py \
        ${INSTANCE_NAME}:~/ --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml up -d --build --force-recreate" --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="mkdir -p models" --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml logs -f" --zone us-east1-d

elif [ "$1" == "reset" ]; then

    echo "Deleting models in ./models/"
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo rm -rf ./models/" --zone us-east1-d
    echo "Deleting caches"
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo rm -rf ./cache/" --zone us-east1-d
    echo "Deleting volumes"
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml down -v" --zone us-east1-d

elif [ "$1" == "download-models" ]; then

    gcloud compute scp ${INSTANCE_NAME}:./models/* ./models/ --recurse --zone us-east1-d

elif [ "$1" == "download-data" ]; then

    gcloud compute scp ${INSTANCE_NAME}:./data/* ./data/ --recurse --zone us-east1-d

elif [ "$1" == "logs" ]; then

    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml logs -f" --zone us-east1-d

fi