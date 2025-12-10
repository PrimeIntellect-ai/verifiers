import verifiers as vf
# Assuming the GymEnv class provided in previous turn is in 'gym_env.py'
from gym_env import GymEnv 
from gem.envs.game_env.wordle import WordleEnv

### Prompt
# GEM relies on regex matching for \boxed{}, so we instruct the model accordingly.
GEM_WORDLE_SYSTEM_PROMPT = """You are a competitive Wordle player.
Your goal is to guess the secret 5-letter word within 20 turns.

In each turn:
1. Think step-by-step about the feedback (G=Green/Correct, Y=Yellow/Wrong Pos, X=Gray/Wrong).
2. Output your final guess inside \\boxed{YOUR_GUESS}.

Example:
<think>
I need to test vowels. "ADIEU" is a good start.
</think>
\\boxed{ADIEU}
"""

### Feedback Function
def wordle_obs_cleaner(obs: str) -> str:
    """
    Optional: Clean GEM's observation if needed.
    GEM obs usually comes as: "At turn X, you guessed Y\nFeedback:..."
    This is already chat-friendly, so we can return it as-is.
    """
    return str(obs)

### Reward Functions
def gem_accumulated_reward(state: vf.State, **kwargs) -> float:
    """
    GEM environments return dense rewards (intermediate milestones) + sparse terminal rewards.
    We simply sum up the rewards returned by the environment step.
    """
    trajectory = state.get("trajectory", [])
    total = sum(float(step.get("reward", 0.0) or 0.0) for step in trajectory)
    return total

def gem_success_bonus(state: vf.State, **kwargs) -> float:
    """
    Additional heuristic to reward strictly winning, based on GEM's success message.
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return 0.0
    
    # Check the last environment response for the success string
    last_step = trajectory[-1]
    env_response = last_step.get("env_response", "")
    
    # Handle if env_response is a list of messages (chat) or string
    content = env_response
    if isinstance(env_response, list) and len(env_response) > 0:
        content = env_response[-1].get("content", "")
        
    if "Congratulations!" in str(content):
        return 1.0
    return 0.0

### Environment Loader
def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 20,
):
    # 1. Define Rubric
    rubric = vf.Rubric()
    # GEM gives internal rewards for progress (milestones), so we use them
    rubric.add_reward_func(gem_accumulated_reward, weight=1.0)
    rubric.add_reward_func(gem_success_bonus, weight=2.0)

    # 2. Action Parser
    # GEM handles parsing internally using regex for \boxed{}.
    # We pass the raw model text (identity function) to env.step().
    action_parser = lambda x: x 

    # 3. Instantiate GymEnv with GEM class
    vf_env = GymEnv(
        env_cls=WordleEnv,
        env_kwargs={
            "word_length": 5,
            "max_turns": 20,
            "only_real_words": True
        },
        action_parser=action_parser,
        obs_to_text=wordle_obs_cleaner,
        rubric=rubric,
        num_train_episodes=num_train_examples,
        system_prompt=GEM_WORDLE_SYSTEM_PROMPT,
        message_type="chat",
        # GymEnv will auto-generate a dummy dataset since GEM randomizes words internally on reset
        auto_dummy_eval=True 
    )
    
    return vf_env