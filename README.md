# deeplearning_tutorial

## Syllabus
<br/>
1. Introduction - setup & components ( perceptrons, relu, cnn )  & MNIST 
<br/>
2. cost function, back propagation & DNN for classification & Transfer learning
<br/>
3. rnn & lstm, training technique ( BN, regularization, dropout, augmentation ) & Tensorboard
<br/>
4. Detection & Segmentation ( Yolo2 experiment & Deeplab v3 experiment )
<br/>
5. RL intro ( DQN & Atari experiment )
<br/>
6. Gym env ( A3C & Navigation experiment )
<br/>
7. Generative Model ( Auto-encoder & GAN & DCGAN ...? )
<br/>
8. Jetson TX2 setup & Test TensorRT
<br/>
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
