build_docker()
{
  echo "building docker "
  docker build -t "sciassit-img" -f docker/Dockerfile .
}

run_docker()
{
  echo "runing docker "
  echo "$FAST_API_PORT"
  docker run -dit --rm -v "$(pwd)":/sciassit -p 80:80 -w /sciassit --name "sciassit-cnt" --tty "sciassit-img"
}

run_service()
{
  echo "running service"
  docker exec -it "sciassit-cnt" uvicorn api:app --host 0.0.0.0 --port 80
}

push_docker()
{
  docker login
}

build_docker
run_docker
run_service

