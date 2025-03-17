echo $CR_PAT | docker login ghcr.io -u nosemeocurreapodo --password-stdin
docker build -t ghcr.io/nosemeocurreapodo/mesh_vo:latest -f Dockerfile .
docker push ghcr.io/nosemeocurreapodo/mesh_vo:latest