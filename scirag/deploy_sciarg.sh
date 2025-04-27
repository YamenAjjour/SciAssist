build_docker_base()
{
  echo "building docker "
  docker build -t "yamenajjour/sciassist-base-img" -f scirag/docker-base/Dockerfile .
}

build_docker()
{
  echo "building docker "
  docker build -t "yamenajjour/sciassist-img" -f scirag/docker/Dockerfile .
}


run_docker()
{
  echo "runing docker "
  echo "$FAST_API_PORT"
  docker run -it --rm --gpus all -v "$(pwd)":/sciassit  -p 80:80 -w /sciassit --name "sciassist-cnt" --tty "yamenajjour/sciassist-img"

}

run_service()
{
  echo "running service"
  docker exec -it "yamenajjour/sciassit-cnt" uvicorn api:app --host 0.0.0.0 --port 80

}

push_docker()
{
  docker login
  docker push "yamenajjour/sciassist-img"
}
build_docker_base
build_docker
push_docker
run_docker

