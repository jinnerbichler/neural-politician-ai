#!/usr/bin/env bash

CLUSTER_NAME="ml-cluster"
ZONE="us-east1-d"
GPU_POOL="gpu-pool"

set -e    # Exits immediately if a command exits with a non-zero status.

# "gcloud config set container/use_v1_api false" must be executed to enable the api

if [ "$1" == "create" ]; then

    gcloud beta container clusters create ${CLUSTER_NAME} \
        --cluster-version=1.9.4-gke.1 --node-version=1.9.4-gke.1 \
        --machine-type=g1-small --preemptible --num-nodes=1 \
        --zone=${ZONE}

elif [ "$1" == "activate" ]; then

    gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE}

elif [ "$1" == "gpu-pool" ]; then

    # https://cloud.google.com/kubernetes-engine/docs/concepts/gpus
    # https://cloud.google.com/compute/docs/gpus/add-gpus
    gcloud beta container node-pools create ${GPU_POOL} \
        --machine-type=n1-standard-2 --preemptible \
        --accelerator type=nvidia-tesla-k80,count=1 \
        --num-nodes=0 --enable-autoscaling --min-nodes=0 --max-nodes=1 \
        --cluster=${CLUSTER_NAME} --zone=${ZONE}

elif [ "$1" == "gpu-pool-daemon" ]; then

    kubectl create -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/k8s-1.9/nvidia-driver-installer/cos/daemonset-preloaded.yaml

elif [ "$1" == "gpu-pool-down" ]; then

    gcloud container clusters resize ${CLUSTER_NAME} --quiet \
        --node-pool ${GPU_POOL} \
        --size 0 --zone ${ZONE}

elif [ "$1" == "gpu-pool-delete" ]; then

    gcloud beta container node-pools delete ${GPU_POOL} --quiet \
        --zone=${ZONE} --cluster=${CLUSTER_NAME}

fi