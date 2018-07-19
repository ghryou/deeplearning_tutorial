# deeplearning_tutorial

## Syllabus
1. Introduction - setup & components ( perceptrons, relu, cnn )  & MNIST
2. cost function, back propagation & DNN for classification & Transfer learning
3. rnn & lstm, training technique ( BN, regularization, dropout, augmentation ) & Tensorboard
4. Detection & Segmentation ( Yolo2 experiment & Deeplab v3 experiment )
5. RL intro ( DQN & Atari experiment )
6. Gym env ( A3C & Navigation experiment )
7. Generative Model ( Auto-encoder & GAN & DCGAN ...? )
8. Jetson TX2 setup & Test TensorRT
9. End-to-end Visuomotor Learning Review

## Docker Setup
### Dockerfile CPU
```
docker run --name pytorch --rm \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e DISPLAY=$DISPLAY \
    --net=host \
    --mount 'type=bind,src=/home/ghryou/Workspace/deeplearning_tutorial,dst=/app' \
    -it pytorch bash

jupyter notebook --allow-root
```

### Dockerfile GPU
```
docker run --name pytorch --rm \
   --runtime=nvidia \
   -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
   -e QT_X11_NO_MITSHM=1 \
   -e DISPLAY=$DISPLAY \
   --net=host \
   --mount 'type=bind,src=/home/naverlabs/DNN_workspace,dst=/app' \
   -it pytorch:cuda2 bash​
```

### Docker Cheat Sheet
```
docker build -t <tag name> -f <Dockerfile path> <Dockerfile directory>
docker images
docker ps -al
docker stop <image id>
docker rmi <image id>
docker system prune
```
