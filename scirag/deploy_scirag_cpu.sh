#!/bin/bash
. ./scirag/.deploy_variables
echo $BASE_IMAGE


build_docker()
{
  echo "building docker "
  docker build -t "$IMAGE" -f scirag/docker_cpu/Dockerfile .
}


run_docker()
{
  echo "runing docker "
  echo "$FAST_API_PORT"
  docker run -it --rm -v "$(pwd)":/sciassit  -p 80:80 -w /sciassit --name "$CONTAINER" --tty "$IMAGE"

}

run_service()
{
  echo "running service"
  docker exec -it "$CONTAINER" all uvicorn api:app --host 0.0.0.0 --port 80

}


push_docker()
{
  docker login
  docker push "$IMAGE"
}

echo "hello"


while getopts "brp" opt; do
echo "while"

echo $opt
case $opt in

b)
  echo "building"
build_docker
;;
p)
  echo "pushing"
push_docker
;;
r)
  echo "running"
run_docker
run_service
;;

:)
      echo "left"
      # This case handles an argument that is present but has no value
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
esac

done
