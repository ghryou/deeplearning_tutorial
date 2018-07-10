# deeplearning_tutorial

## Docker Setup
```
docker run --name pytorch --rm \
    --mount 'type=bind,src=/home/gryou/Workspace/deeplearning_tutorial,dst=/app' \
    -it pytorch bash
```
