#!/bin/bash
. ./scirag/.deploy_variables
echo $BASE_IMAGE


build_docker()
{
  echo "building docker "
  docker build -t "$IMAGE" -f scirag/docker/Dockerfile .

}


run_docker()
{
  echo "runing docker "
  echo "$FAST_API_PORT"
  docker run -it --rm --gpus all -v "$(pwd)":/sciassit  -p 80:80 -w /sciassit --name "$CONTAINER" --tty "$IMAGE"

}

run_service()
{
  echo "running service"
  docker exec -it "$CONTAINER" --gpus all uvicorn api:app --host 0.0.0.0 --port 80

}


push_docker()
{
  docker login
  docker push "$IMAGE"
}

build_docker
push_docker
run_docker

