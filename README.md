# deeplearning_tutorial

## Docker Setup
```
docker run --name pytorch --rm \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e DISPLAY=$DISPLAY \
    --net=host \
    --mount 'type=bind,src=/home/gryou/Workspace/deeplearning_tutorial,dst=/app' \
    -it pytorch bash

jupyter notebook --allow-root
```
