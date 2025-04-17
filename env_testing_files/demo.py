import gym
import minerl
import logging
import imageio
logging.disable(logging.ERROR)

# env = gym.make('MineRLBasaltFindCave-v0')
# env = gym.make("MineRLTreechop-v0")
env = gym.make("MineRLObtainDiamondShovel-v0")

obs = env.reset()

done = False
frames = []

while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    # frame = env.render(mode = 'rgb_array')
    # frames.append(frame)
    env.render()
    

# imageio.mimsave('simulation.mp4', frames, fps=5)
# import logging
# import gym
# from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

# import coloredlogs
# coloredlogs.install(logging.DEBUG)

# def test_turn(resolution):
#     #env = HumanSurvival(resolution=resolution).make()
#     #env = gym.make("MineRLBasaltBuildVillageHouse-v0")
#     env = gym.make("MineRLObtainDiamondShovel-v0")
#     #env = gym.make("MineRLBasaltFindCave-v0")
#     env.reset()
#     _, _, _, info = env.step(env.action_space.noop())
#     N = 100
#     for i in range(N):
#         ac = env.action_space.noop()
#         ac['camera'] = [0.0, 360 / N]
#         _, _, _, info = env.step(ac)
#         env.render()
#     env.close()

# if __name__ == '__main__':
#     test_turn((640, 360))


