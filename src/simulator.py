import cv2
import numpy as np
import random

simulator = {"width":31, "height":31, "center":15, "resol":3}
map_param = {"width":100, "height":100, "center":50, "resol":1, "scale":5, "Map.data.obstacle": 200, "Map.data.ball": 1}
# walls_samples = [[1.2,2.0],[1.4,2.0],[1.6,2.0],[1.2,4.0],[1.4,4.0],[1.6,4.0],[1.2,-2.0],[1.4,-2.0],[1.6,-2.0],[1.2,-4.0],[1.4,-4.0],[1.6,-4.0]]
walls_samples = [[1.5,-30.0],[30.4,0.7],[-30.4,-0.7]]

camera_fov = 120
ball_blind_ratio = 1/np.tan(camera_fov/2*np.pi/180)
ball_blind_bias = 5

reward_region_x = 2
reward_region_y = [0,1,2,3]

trans_scale = int(simulator["resol"]/map_param["resol"])
rot_scale = 20

debug_scale = 10
debug_scale_gray = 3

max_iter = 99

class TTbotGym:
    def __init__(self, debug_flag=False, mlp_flag=False, test_flag=False, state_blink=True, state_inaccurate=True):
        self.frame = np.zeros((simulator["height"],simulator["width"],1), np.uint8)
        self.frame_gray = np.zeros((simulator["height"]*debug_scale_gray,simulator["width"]*debug_scale_gray,1), np.uint8)
        self.balls = []
        self.balls_prev = []
        self.obstacles = []
        self.episode_rewards = []
        self.score = 0
        self.iter = 0
        self.done = False

        self.write_flag = False
        self.debug_flag = debug_flag
        self.mlp_flag = mlp_flag
        self.test_flag = test_flag
        self.ball_inscreen_flag = 0
        self.state_blink = state_blink
        self.state_inaccurate = state_inaccurate

        ############################################################
        # Flag Setups
        # debug_flag : Display colored map for demonstration
        # mlp_flag : If true, step() returns small 31x31 gray scale image for Multi Layer Perception.
        #            If false, step() returns 93x93 gray scale image for Convolutional Neural Network
        # test_flag : Whether it is training phase or test phase. During the test phase, the simulator saves every demonstration in form of video
        # write_flag : If true, it records videos
        # state_blink : Simulate target's detection error
        # state_inaccurate : Simulate target's position estimation error
        ############################################################

        ## DQN parameters
        if self.mlp_flag:
            self.observation_space = self.frame.copy()
        else:
            self.observation_space = self.frame_gray.copy()

        self.action_space = np.array(range(10))


        # self.reset(max_balls=20, max_walls=3)
        return

    def reset(self, max_balls=20, max_walls=2):
        self.frame = np.zeros((simulator["height"],simulator["width"],1), np.uint8)
        self.frame_gray = np.zeros((simulator["height"]*debug_scale_gray,simulator["width"]*debug_scale_gray,1), np.uint8)
        self.balls = []
        self.balls_prev = []
        self.obstacles = []
        self.score = 0
        self.iter = 0
        self.done = False
        self.write_flag = False
        self.ball_inscreen_flag = 0

        if len(self.episode_rewards)%5000 == 0 and not self.test_flag:
            self.write_flag = True
            out_directory = "data/video/tt.video."+format(len(self.episode_rewards)/5000,"06")+".mp4"

        if self.test_flag:
            self.write_flag = True
            out_directory = "data/video_test/tt.video."+format(len(self.episode_rewards),"06")+".mp4"

        if self.write_flag:
            codec = cv2.VideoWriter_fourcc(*'H264')
            fps = 10
            self.video = cv2.VideoWriter(out_directory, codec, fps, (simulator["width"]*debug_scale,simulator["height"]*debug_scale))

        num_walls = int((max_walls+1)*random.random())
        # walls_sampled = random.sample(walls_samples, num_walls)
        rand_direction = random.random()
        if rand_direction >= 0.666:
            walls_sampled = [[random.random()+0.7,-10*random.random()-20],[-random.random()-0.2,-10*random.random()-20],\
                        [10*random.random()+20,0.5*random.random()+0.4],[-10*random.random()-20,-0.5*random.random()-0.4]]
        elif rand_direction >= 0.333:
            walls_sampled = [[random.random()+0.7,10*random.random()+20],[-random.random()-0.2,10*random.random()+20],\
                        [-10*random.random()-20,0.5*random.random()+0.4],[10*random.random()+20,-0.5*random.random()-0.4]]
        else:
            walls_sampled = []

        obstacles_temp = []
        for wall in walls_sampled:
            if abs(wall[1]) >= abs(wall[0]):
                point_start = -2.0*map_param["center"]
                point_end = 2.0*map_param["center"]
                unit = (point_end-point_start)/200

                for i in range(200):
                    cy = (point_start + unit*i)
                    cx = (wall[0]*(map_param["center"]-(cy/wall[1])))
                    obstacles_temp.append([cx,cy])
            else:
                point_start = -1.0*map_param["center"]
                point_end = 3.0*map_param["center"]
                unit = (point_end-point_start)/200

                for i in range(200):
                    cx = (point_start + unit*i)
                    cy = (wall[1]*(map_param["center"]-(cx/wall[0])))
                    obstacles_temp.append([cx,cy])

        for obstacle in obstacles_temp:
            cx = obstacle[0]
            cy = obstacle[1]
            insert = True
            for wall in walls_sampled:
                if cx/wall[0] + cy/wall[1] > map_param["center"]:
                    insert = False
            if insert:
                self.obstacles.append([cx,cy])

        for i in range(max_balls):
            cx = int(1.0*random.random()*(map_param["height"]-2*trans_scale)+2*trans_scale)
            cy = int(1.0*random.random()*map_param["width"]) - map_param["center"]
            insert = True
            for wall in walls_sampled:
                if cx/wall[0] + cy/wall[1] >= map_param["center"]:
                    insert = False
            if insert:
                self.balls.append([cx,cy])

        ## For Test - Put every ball on a single horizontal line
        # for i in range(max_balls):
        #     cx = int(0.3*(map_param["height"]-2*trans_scale)+2*trans_scale)
        #     cy = int(0.4*(int(1.0*i/max_balls*map_param["width"]) - map_param["center"]))
        #     insert = True
        #     for wall in walls_sampled:
        #         if cx/wall[0] + cy/wall[1] >= map_param["center"]:
        #             insert = False
        #     if insert:
        #         self.balls.append([cx,cy])
        ###########################

        self.frame, self.frame_gray = self.draw_state()

        if self.mlp_flag:
            return self.frame
        else:
            return self.frame_gray

    def check_window_state(self, cx, cy):
        inscreen = True
        if cx < 0 or cx >= simulator["width"]:
            inscreen = False
        if cy < 0 or cy >= simulator["height"]:
            inscreen = False
        return inscreen

    def draw_debug_frame(self, frame):
        frame_debug = np.zeros((simulator["height"]*debug_scale,simulator["width"]*debug_scale,3), np.uint8)
        for i in range(simulator["width"]):
            for j in range(simulator["height"]):
                if frame[i][j] == map_param["Map.data.obstacle"]:
                    cv2.rectangle(frame_debug,(i*debug_scale,j*debug_scale),((i+1)*debug_scale-1,(j+1)*debug_scale-1),(255,255,0),-1)
                if frame[i][j] == map_param["Map.data.ball"]:
                    cv2.rectangle(frame_debug,(i*debug_scale,j*debug_scale),((i+1)*debug_scale-1,(j+1)*debug_scale-1),(0,255,0),-1)

        cv2.rectangle(frame_debug,(simulator["center"]*debug_scale-1,(simulator["height"]-1)*debug_scale+1),\
                    ((simulator["center"]+1)*debug_scale,simulator["height"]*debug_scale-1),(255,0,0),-1)

        for i in range(1,simulator["width"]):
            cv2.line(frame_debug,(i*debug_scale,0),(i*debug_scale,simulator["height"]*debug_scale-1),(128,128,128),1)
            cv2.line(frame_debug,(0,i*debug_scale),(simulator["width"]*debug_scale-1,i*debug_scale),(128,128,128),1)

        cv2.line(frame_debug,((simulator["center"]+ball_blind_bias)*debug_scale,simulator["height"]*debug_scale-1),\
                            (simulator["width"]*debug_scale-1,(simulator["height"]-int(ball_blind_ratio*(simulator["center"]-1-ball_blind_bias)))*debug_scale),(128,128,128),1)
        cv2.line(frame_debug,((simulator["center"]-ball_blind_bias+1)*debug_scale,simulator["height"]*debug_scale-1),\
                            (0,(simulator["height"]-int(ball_blind_ratio*(simulator["center"]-1-ball_blind_bias)))*debug_scale),(128,128,128),1)

        cv2.rectangle(frame_debug,((simulator["center"]-2)*debug_scale-1,(simulator["height"]-2)*debug_scale+1),\
                    ((simulator["center"]+3)*debug_scale,simulator["height"]*debug_scale-1),(0,0,255),2)

        cv2.putText(frame_debug,"Score "+str(self.score), (int(simulator["width"]*debug_scale*0.65),int(simulator["width"]*debug_scale*0.05)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
        cv2.putText(frame_debug,"Step "+str(self.iter), (int(simulator["width"]*debug_scale*0.05),int(simulator["width"]*debug_scale*0.05)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))

        return frame_debug

    def draw_state(self):
        frame = np.zeros((simulator["height"],simulator["width"],1), np.uint8)
        frame_gray = np.zeros((simulator["height"]*debug_scale_gray,simulator["width"]*debug_scale_gray,1), np.uint8)

        for obstacle in self.obstacles:
            cx = simulator["center"] - int(round(1.0*obstacle[1]/trans_scale))
            cy = simulator["height"] - 1 - int(round(1.0*obstacle[0]/trans_scale))
            if self.check_window_state(cx, cy):
                frame[cx][cy] = map_param["Map.data.obstacle"]
        for ball in self.balls:
            if self.state_blink == False or random.random() > (0.3 + ball[0]/3.0/map_param["center"]):
                if ball[0] >= int(ball_blind_ratio*(abs(1.0*ball[1])-ball_blind_bias)):
                    ball_x = ball[0]
                    ball_y = ball[1]
                    if self.state_inaccurate:
                        ball_x = ball_x + random.random()*map_param["center"]*(0.1*ball_x*ball_x/map_param["center"]/map_param["center"] - 0.05)
                        ball_y = ball_y + random.random()*map_param["center"]*(0.1*ball_x*ball_x/map_param["center"]/map_param["center"] - 0.05)

                    cx = simulator["center"] - int(round(1.0*ball_y/trans_scale))
                    cy = simulator["height"] - 1 - int(round(1.0*ball_x/trans_scale))
                    if self.check_window_state(cx, cy):
                        frame[cx][cy] = map_param["Map.data.ball"]
        frame[simulator["center"]][simulator["height"]-1] = 255

        if not self.mlp_flag:
            gray_color = {"ball":255, "wall":100, "robot":200, "robot_padding":150}
            for i in range(simulator["width"]):
                for j in range(simulator["height"]):
                    if frame[i][j] == map_param["Map.data.obstacle"]:
                        cv2.rectangle(frame_gray,(i*debug_scale_gray,j*debug_scale_gray),((i+1)*debug_scale_gray-1,(j+1)*debug_scale_gray-1),gray_color["wall"],-1)
                    if frame[i][j] == map_param["Map.data.ball"]:
                        cv2.rectangle(frame_gray,(i*debug_scale_gray,j*debug_scale_gray),((i+1)*debug_scale_gray-1,(j+1)*debug_scale_gray-1),gray_color["ball"],-1)

            cv2.rectangle(frame_gray,((simulator["center"]-2)*debug_scale_gray-1,(simulator["height"]-2)*debug_scale_gray+1),\
                        ((simulator["center"]+3)*debug_scale_gray,simulator["height"]*debug_scale_gray-1),gray_color["robot_padding"],-1)
            cv2.rectangle(frame_gray,(simulator["center"]*debug_scale_gray-1,(simulator["height"]-1)*debug_scale_gray+1),\
                        ((simulator["center"]+1)*debug_scale_gray,simulator["height"]*debug_scale_gray-1),gray_color["robot"],-1)

        return frame, frame_gray

    def get_reward(self, action):
        reward = 0
        balls_temp = []
        for i, ball in enumerate(self.balls):
            cx = int(round(1.0*ball[0]/trans_scale))
            cy = int(round(abs(1.0*ball[1]/trans_scale)))
            if  cx < reward_region_x and cx >= 0 and ball[0] >= int(ball_blind_ratio*(abs(1.0*ball[1])-ball_blind_bias)):
                if cy <= reward_region_y[0]:
                    reward = reward + 3
                elif cy <= reward_region_y[1]:
                    reward = reward + 2
                elif cy <= reward_region_y[2]:
                    reward = reward + 1
                    if len(self.balls_prev) > 0:
                        if int(round(1.0*self.balls_prev[i][0]/trans_scale)) < reward_region_x:
                            reward = reward - 2
                else:
                    balls_temp.append(ball)
            else:
                balls_temp.append(ball)

        balls_inscreen = []
        for ball in balls_temp:
            if ball[0] >= ball_blind_ratio * (abs(1.0*ball[1]) - ball_blind_bias)\
                and abs(1.0*ball[1]) <= map_param["center"] and abs(1.0*ball[0]) < map_param["height"]:
                balls_inscreen.append(ball)

        self.balls = balls_temp
        if self.debug_flag:
            print("balls length : "+str(len(balls_temp))+"  score : "+str(self.score)+"  screen_flag : "+str(self.ball_inscreen_flag))

        if action in range(10):
            if len(balls_inscreen) == 0:
                self.ball_inscreen_flag = self.ball_inscreen_flag + 1
            else:
                self.ball_inscreen_flag = 0

        if len(balls_temp) == 0 or self.iter > max_iter or self.ball_inscreen_flag >= 10:
            self.done = True

        if self.done:
            self.episode_rewards.append(self.score)
            if self.write_flag:
                self.video.release()
                print("video saved")

        if action == -1:
            return -1
        else:
            return reward

    def step(self, action):
        if action in range(10):
            self.iter = self.iter + 1

        del_x, del_y, rot = 0, 0, 0

        if action == 0: # forward
            del_x, del_y = -1, 0
        elif action == 1: # forward right
            del_x, del_y = -1, 1
        elif action == 2: # right
            del_x, del_y = 0, 1
        elif action == 3: # backward right
            del_x, del_y = 1, 1
        elif action == 4: # backward
            del_x, del_y = 1, 0
        elif action == 5: # bacward left
            del_x, del_y = 1, -1
        elif action == 6: # left
            del_x, del_y = 0, -1
        elif action == 7: # forward left
            del_x, del_y = -1, -1
        elif action == 8: # turn left
            rot = -1
        elif action == 9: # turn right
            rot = 1
        else:
            del_x, del_y, rot_x = 0, 0, 0

        balls_temp = []
        obstacles_temp = []

        del_x = del_x * trans_scale
        del_y = del_y * trans_scale


        # Update the positions of balls and obstacles
        if len(self.balls) > 0:
            balls_temp = np.add(self.balls, [del_x,del_y])

        if len(self.obstacles) > 0:
            obstacles_temp = np.add(self.obstacles, [del_x,del_y])

        if action == 8 or action == 9:
            if len(self.obstacles) > 0 and len(balls_temp) > 0:
                points = np.concatenate((balls_temp, obstacles_temp))
            else:
                points = np.array(balls_temp)

            if points.size > 0:
                points = points.reshape(-1,2)
                theta = rot_scale*rot*np.pi/180
                theta_0 = np.arctan2(points.T[1],points.T[0])

                ball_dist = np.linalg.norm(points, axis=1)
                rot_delta_unit_x = np.subtract(np.cos(theta_0), np.cos(np.add(theta_0,theta)))
                rot_delta_unit_y = np.subtract(np.sin(theta_0), np.sin(np.add(theta_0,theta)))
                rot_delta_unit = np.concatenate((rot_delta_unit_x.reshape(-1,1),rot_delta_unit_y.reshape(-1,1)),axis=1)
                ball_dist = np.concatenate((ball_dist.reshape(-1,1),ball_dist.reshape(-1,1)),axis=1)
                rot_delta = np.multiply(ball_dist, rot_delta_unit)
                points = np.subtract(points, rot_delta)

                balls_temp = points[0:len(self.balls)]
                obstacles_temp = points[len(self.balls):]

        # Check collision
        enable_move = True
        for obstacle in obstacles_temp:
            # if int(abs(1.0*obstacle[0]/trans_scale)) <= 0 and int(abs(1.0*obstacle[1]/trans_scale)) <= 0:
            if abs(1.0*obstacle[0]) < 2.0 and abs(1.0*obstacle[1]) < 2.0:
                enable_move = False

        # Update observation state and calculate reward
        reward = 0
        if enable_move:
            self.balls = balls_temp
            reward = self.get_reward(action)
            self.obstacles = obstacles_temp
            self.frame, self.frame_gray = self.draw_state()
            self.balls_prev = self.balls
        else:
            reward = self.get_reward(-1)

        self.score = self.score + reward

        # Draw debug frame to record video
        if self.write_flag:
            frame_debug = self.draw_debug_frame(self.frame)
            self.video.write(frame_debug)

        # Draw debug frame to display debug map
        if self.debug_flag:
            frame_debug = self.draw_debug_frame(self.frame)
            cv2.imshow("frame_debug", frame_debug)
            cv2.imshow("frame_debug_gray", self.frame_gray)

        if self.mlp_flag:
            return self.frame, reward, self.done
        else:
            return self.frame_gray, reward, self.done

    def render(self):
        frame_debug = self.draw_debug_frame(self.frame)
        return frame_debug

    def get_total_steps(self):
        return self.iter

    def get_episode_rewards(self):
        return self.episode_rewards

    def action_space_sample(self):
        index = int(1.0*random.random()*10)
        return self.action_space[index]

if __name__ == '__main__':
    env = TTbotGym(debug_flag=True, mlp_flag=False, test_flag=False, state_blink=True, state_inaccurate=True)
    env.reset()

    action = -1
    while(1):
        env.step(action)
        key = cv2.waitKey(300)&0xFF
        action = -1
        if key == ord('q') or env.done == True:
            break;
        elif key == ord('w'):
            action = 0
        elif key == ord('d'):
            action = 2
        elif key == ord('s'):
            action = 4
        elif key == ord('a'):
            action = 6
        elif key == ord('z'):
            action = 8
        elif key == ord('c'):
            action = 9

    print("shutdown")
    cv2.destroyAllWindows()
