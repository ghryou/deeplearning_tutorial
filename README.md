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
```
docker run --name pytorch --rm \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e DISPLAY=$DISPLAY \
    --net=host \
    --mount 'type=bind,src=/home/gryou/Workspace/deeplearning_tutorial,dst=/app' \
    -it pytorch:ubuntu bash

jupyter notebook --allow-root
```
