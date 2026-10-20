# Assignment 1: Reinforcement Learning

**Name:** *pending*  
**GitHub username:** *pending*  
**Repository name:** *pending*  
**Link to the GitHub repository:** *pending*  

## Instructions for running the code

> Instructions of how to run your code.

*Answer pending.*

## 2. Value and Policy Iteration (30 points)

### Solution overview

> Provide a description of your solution and any corresponding plots.

*Answer pending.*

### MDP formulation

> Model the shuttle dispatching problem as a Markov Decision Process, given K = 15, c_f = 100, c_h = 2 and A_t ~ Unif{1, 2, 3, 4, 5}. The number of people in the station cannot exceed 200; use discount rate γ = 0.95.

*Answer pending.*

### (a) Enumeration, T = 500

> Solve using enumeration (with time period T = 500). Plot the optimal value function at time 0 versus number of customers waiting.

*Figure 1 (pending). Section 2 (a), enumeration with T = 500: optimal value function at time 0 versus number of customers waiting. Generate it with `python run.py dp --method enumeration`.*

*Answer pending.*

### (b) Value iteration, T = ∞

> Solve using value iteration (T = ∞). Plot optimal value function versus number of customers waiting.

*Figure 2 (pending). Section 2 (b), value iteration (T = ∞): optimal value function versus number of customers waiting. Generate it with `python run.py dp --method value_iteration`.*

*Answer pending.*

### (c) Policy iteration

> Solve using policy iteration. Plot optimal policy versus number of customers waiting.

*Figure 3 (pending). Section 2 (c), policy iteration: optimal policy versus number of customers waiting. Generate it with `python run.py dp --method policy_iteration`.*

*Answer pending.*

### Assumptions

> If you make any other assumption, please state it in your report.

*Answer pending.*

### Edge cases

> What edge cases may arise from the problem and how did you handle them? Document the edge cases of the problem that you considered and explain how you ensured that your solution has handled them correctly.

*Answer pending.*

### Tests for validating the correctness of your algorithm

> Beyond asking AI, how do you know that your algorithm is working correctly? Test whether your implementation is behaving as predicted. For example, you may change certain parameters or the structure of the problem and see if the output matches your expectation. You may also reduce the stochasticity of the environment or policy so that the process is close to deterministic. You may also find it useful to isolate the first 10 steps and observe how the states, values, rewards, and policy interact at each step. Provide directions and documentation for how to execute validation and test experiments (what does each test do, how does it validate, etc). Provide quantitative and convincing evidence. (Not unit tests or basic bug fixes, such as checking the data format.)

*No validation results yet. Add experiments to tests/ and run `python run.py tests --section dp`.*

*Answer pending.*

### Additional validation with 3 more months

> Given an additional 3 months (extended period), what other validation tests and experiments would you perform?

*Answer pending.*

## 3. Policy Gradient: Part 1 (40 points)

### Solution overview

> Using PyTorch, implement the basic policy gradient algorithm: train a simple neural network that models the policy and takes actions based on the provided states. Set the discounted reward over an episode as your objective function and use Adam as your gradient optimizer with learning rate 0.01 (if you use a different learning rate because training is more stable, mention it). Provide a description of your solution and any corresponding plots.

*Answer pending.*

### (a) CartPole-v1, discount factor 0.95

> Plot the episode reward against the number of training episodes, overlaid with the moving average of the rewards over 100 episodes. When you finish training, roll out 500 episodes using your trained model. Plot the histogram of the 500 episode rewards and report the mean and the standard deviation.

*Figure 4 (pending). Part 1 (a), CartPole-v1: episode reward versus training episode, with the 100-episode moving average. Generate it with `python run.py pg --part 1 --env cartpole`.*

*Figure 5 (pending). Part 1 (a), CartPole-v1: histogram of the rewards of 500 episodes rolled out with the trained policy. Generate it with `python run.py pg --part 1 --env cartpole`.*

**Evaluation over 500 episodes:** mean and standard deviation pending.

*Answer pending.*

### (b) Pong-v5, discount factor 0.99

> Use frames of the game as input to a neural net which models the policy. The game has six actions [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE], but you should train a network that chooses the best action only between [RIGHT, LEFT] (actions 2 and 3). You may pre-process the frames with the provided preprocess function. Use the specified environment version (Pong-v5); on deepdish, use a single GPU card. Plot the episode reward against the number of training episodes, overlaid with the moving average of the rewards over 100 episodes. When you finish training, roll out 500 episodes using your trained model. Plot the histogram of the 500 episode rewards and report the mean and the standard deviation.

*Figure 6 (pending). Part 1 (b), Pong-v5: episode reward versus training episode, with the 100-episode moving average. Generate it with `python run.py pg --part 1 --env pong`.*

*Figure 7 (pending). Part 1 (b), Pong-v5: histogram of the rewards of 500 episodes rolled out with the trained policy. Generate it with `python run.py pg --part 1 --env pong`.*

**Evaluation over 500 episodes:** mean and standard deviation pending.

*Answer pending.*

### Edge cases

> What edge cases may arise from the problem and how did you handle them? Document the edge cases of the problem that you considered and explain how you ensured that your solution has handled them correctly.

*Answer pending.*

### Tests for validating the correctness of your algorithm

> Beyond asking AI, how do you know that your algorithm is working correctly? Test whether your implementation is behaving as predicted. For example, you may change certain parameters or the structure of the problem and see if the output matches your expectation. You may also reduce the stochasticity of the environment or policy so that the process is close to deterministic. You may also find it useful to isolate the first 10 steps and observe how the states, values, rewards, and policy interact at each step. Provide directions and documentation for how to execute validation and test experiments (what does each test do, how does it validate, etc). Provide quantitative and convincing evidence. (Not unit tests or basic bug fixes, such as checking the data format.)

*No validation results yet. Add experiments to tests/ and run `python run.py tests --section pg1`.*

*Answer pending.*

### Additional validation with 3 more months

> Given an additional 3 months (extended period), what other validation tests and experiments would you perform?

*Answer pending.*

## Policy Gradient: Part 2 (20 points)

### Baseline: choice and implementation

> The training suffers from high variance of the rewards in the sampled episodes; using a baseline helps to reduce the variance. Repeat the experiments of Part 1 using a constant or non-constant baseline of your choice. Explain which baseline you have chosen and how you implemented it.

*Answer pending.*

### Solution overview

> Provide a description of your solution and any corresponding plots.

*Answer pending.*

### (a) CartPole-v1 with baseline

> Repeat Part 1 (a) with your baseline. Plot the episode reward against the number of training episodes, overlaid with the moving average of the rewards over 100 episodes. When you finish training, roll out 500 episodes using your trained model. Plot the histogram of the 500 episode rewards and report the mean and the standard deviation.

*Figure 8 (pending). Part 2 (a), CartPole-v1: episode reward versus training episode, with the 100-episode moving average. Generate it with `python run.py pg --part 2 --env cartpole`.*

*Figure 9 (pending). Part 2 (a), CartPole-v1: histogram of the rewards of 500 episodes rolled out with the trained policy. Generate it with `python run.py pg --part 2 --env cartpole`.*

**Evaluation over 500 episodes:** mean and standard deviation pending.

*Answer pending.*

### (b) Pong-v5 with baseline

> Repeat Part 1 (b) with your baseline. Plot the episode reward against the number of training episodes, overlaid with the moving average of the rewards over 100 episodes. When you finish training, roll out 500 episodes using your trained model. Plot the histogram of the 500 episode rewards and report the mean and the standard deviation.

*Figure 10 (pending). Part 2 (b), Pong-v5: episode reward versus training episode, with the 100-episode moving average. Generate it with `python run.py pg --part 2 --env pong`.*

*Figure 11 (pending). Part 2 (b), Pong-v5: histogram of the rewards of 500 episodes rolled out with the trained policy. Generate it with `python run.py pg --part 2 --env pong`.*

**Evaluation over 500 episodes:** mean and standard deviation pending.

*Answer pending.*

### Edge cases

> What edge cases may arise from the problem and how did you handle them? Document the edge cases of the problem that you considered and explain how you ensured that your solution has handled them correctly.

*Answer pending.*

### Tests for validating the correctness of your algorithm

> Beyond asking AI, how do you know that your algorithm is working correctly? Test whether your implementation is behaving as predicted. For example, you may change certain parameters or the structure of the problem and see if the output matches your expectation. You may also reduce the stochasticity of the environment or policy so that the process is close to deterministic. You may also find it useful to isolate the first 10 steps and observe how the states, values, rewards, and policy interact at each step. Provide directions and documentation for how to execute validation and test experiments (what does each test do, how does it validate, etc). Provide quantitative and convincing evidence. (Not unit tests or basic bug fixes, such as checking the data format.)

*No validation results yet. Add experiments to tests/ and run `python run.py tests --section pg2`.*

*Answer pending.*

### Additional validation with 3 more months

> Given an additional 3 months (extended period), what other validation tests and experiments would you perform?

*Answer pending.*
