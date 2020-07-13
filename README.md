# ITER_KER_GER

## Description

This repo refers to the paper [*Invariant Transform Experience Replay: Data Augmentation for Deep Reinforcement Learning*](https://arxiv.org/abs/1909.10707#). This repo would be open source once our paper gets accepted.

Deep reinforcement learning (DRL) is a promising approach for adaptive robot control, but its current application to robotics is currently hindered by high sample requirements. We propose two novel data augmentation techniques for DRL based on invariant transformations of trajectories in order to reuse more efficiently observed interaction. The first one called Kaleidoscope Experience Replay exploits reflectional symmetries, while the second called Goal-augmented Experience Replay takes advantage of lax goal definitions. In the Fetch tasks from OpenAI Gym, our experimental results show a large increase in learning speed.

And this repo is built on top of [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines) and [OpenAI Gym](https://github.com/openai/gym). 

## Installation

This implementation requires the installation of the [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines) module (commit version ```2bca79```). 
```
git clone https://github.com/openai/baselines.git
cd baselines
```
Install baselines package (If you meet any problem during the installation, please click [here](https://github.com/openai/baselines/tree/master/baselines)).
```
pip install -e .
```
After the installation, please create a new folder for this repo and go inside.
```
mkdir ITER_KER_GER && cd $_
```
Download all the codes held in this repo.
```
git clone https://github.com/birlrobotics/ITER_KER_GER.git
```
Finally, please copy the files held in folder `ITER_KER_GER/her` and paste into `baselines/baselines/` to overwrite the vanilla `HER` file.
```
cp -rf her ~/baselines/baselines/
```
## Usage

To reproduce the best result in our paper, please run :
```
python -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=1e6 --n_cycles=100 --save_path=/home/user/policies/her/iter --log_path=/home/user/log_data/her/iter --before_GER_minibatch_size=256 --n_KER=8 --n_GER=4
```

options include:
* `--num_cpu`: Number of workers(threads/cpus). The results in our paper just used 1 worker in order to show the significant improvements in learning speed. The [original HER paper](https://arxiv.org/abs/1802.09464) presents this HER implementation. (**Please note that as the HER's author said, running the code with different cpus is NOT equivalent. For more information about this issue, please check [here](https://github.com/openai/baselines/issues/314).**)
* `--env`: To specify the experimental environment in each run. Possible choices are *FetchPickAndPlace-v1, FetchSlide-v1, FetchPush-v1*. (There will be more choices on Baxter robot in the near future, please keep watching on our repo :). )
* `--before_GER_minibatch_size`: To specify the original minibatch size.
* `--n_KER`: To specify the hyperparameter of KER. More specifically, it is to specify how many reflectional planes you would like to augment the samples. For more information, please checkout our [Paper](https://arxiv.org/abs/1909.10707#).
* `--n_GER`: To specify the hyperparameter of GER. Specifically, it is to specify how many transitions' goals you would like to augment. For more information, please checkout our [Paper](https://arxiv.org/abs/1909.10707#).
* `--log_path`: To specify the log file saved path.
* `--save_path`: To specify the policy parameters saved path.

## Loading and visualizing models

This [page](https://github.com/openai/baselines#saving-loading-and-visualizing-models) from OpenAI Baselines has a good indicaition on loading and visualizing models.

## Training Environment

<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/task_ob_without_ob-crop_00.jpg"></center>
<div class="thecap" align="middle"><b>Fig. 1 Testing robotic tasks: Fetch Pushing, Sliding, and Pick-and-Place without obstacles (each column left) and with obstacles (each column right).</b></div>
</div>


## Results

ITER greatly improves the robot's generalization ability by augmenting the observed transition samples with KER and GER, leading to a highly efficient learning process. The learning curves (the Testing Success Rate vs Epoch) plotted below show the significant improvements in three robotic tasks learning with or without obstacles:


<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/best_combination-crop_00.jpg"   width="770" height="258" /></center>
<div class="thecap" align="middle"><b>Fig. 2 Training results for aforementioned robotic tasks without obstacles constrasting between training with and without ITER.</b></div>
</div>

<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/plot_3tasks_obstacle_best_vanilla-crop_00.jpg"   width="770" height="258" /></center>
<div class="thecap" align="middle"><b>Fig. 3 Training results for aforementioned robotic tasks with obstacles constrasting between training with and without ITER.</b></div>
</div>

For more experimental results please read our [Paper](https://arxiv.org/abs/1909.10707#).

## Quick Visulization on Learning Results

You can visulize the testing results during training with [TensorBoard](https://www.tensorflow.org/tensorboard/get_started). _SummaryWriter_ saves the testing result after each epoch in the running directory. You can open a new terminal with that directory and run 
```
tensorboard --logdir ~/
```

## Learned performance with ITER in a More Complex Dynamical Environment

Our method preserves any contact that may occur between the robot and any object it may encounter (table included) as long as a symmetry is applied to all the objects and obstacles in the robot's workspace. Therefore, our approach also works in any contact-rich robotic task (a more complex dynamical environment), including problems where some obstacles may limit the movements of objects or the robot. When the poses of obstacles are observed in each state but not fixed across episodes, the agent can learn the effects of contact. For example, the agent can avoid obstacles or leverage contact to reach a goal (e.g. in the pushing task it may learn to push an object and let the obstacle stop the moving object). 

The following gifs show the comparisons of learned behaviors with HER and ITER.


<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/push_ob1.gif"  width="480" height="260" /></center>
<div class="thecap" align="middle"><b>Learned behaviors at training epoch 80 in Pushing task.</b></div>
</div>



<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/slide_ob1.gif"  width="480" height="260" /></center>
<div class="thecap" align="middle"><b>Learned behaviors at training epoch 100 in Sliding task.</b></div>
</div>

<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/pnp_ob1.gif"  width="480" height="260" /></center>
<div class="thecap" align="middle"><b>Learned behaviors at training epoch 230 in Pick-and-Place task.</b></div>
</div>

## Deployment on a Physical Robot

We also applied a well-trained policy with ITER to a real Baxter robot. To do that, we first trained a pick-and-place policy in simulation ([Baxter in Gym](https://github.com/huangjiancong1/gym_baxter)). Then we transfer it to the real one, and the object pose is detected by using [ALVAR](http://wiki.ros.org/ar_track_alvar/) (more information is in [Appendix](http://www.juanrojas.net/wp-content/uploads/2020/06/2020RAL_ITER_supplement.pdf)).

<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/virtual_baxter.gif"  width="720" height="390" /></center>
<div class="thecap" align="middle"><b>A vertual Baxter robot is trained in the pick-and-place task with ITER (the goal is located at the red ball)</b></div>
</div>

<div class="imgcap" align="middle">
<center><img src="https://github.com/YijiongLin/figs_show/blob/master/real_baxter.gif"  width="720" height="390" /></center>
<div class="thecap" align="middle"><b>A real Baxter robot running a pick-and-place policy trained via ITER (the goal is located at the orange rectangle)</b></div>
</div>

## More Information

For more information please check:
1. [Website Blog](http://www.juanrojas.net/iter/)
2. [Paper](https://arxiv.org/abs/1909.10707#)
3. Video: [Youtube](https://youtu.be/Ac3c_xs7pJ8), [Youku](https://v.youku.com/v_show/id_XNDcwNDQ5MDk5Mg==.html)
4. [Appendix](http://www.juanrojas.net/wp-content/uploads/2020/06/2020RAL_ITER_supplement.pdf)

## Credits

`ITER_KER_GER` is maintained by the BIRL Intelligent Manipulation group. Contributors include:

- Yijiong Lin (Bourne), 2111701025@mail2.gdut.edu.cn （Currently looking for 2020 fall Ph.D/MRes Program）
- Jiancong Huang (Jim), huangjiancong863@gmail.com （Currently looking for 2021 fall Ph.D/MRes Program）
