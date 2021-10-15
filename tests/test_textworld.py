import gym
import textworld.gym
from textworld.generator import compile_game
from textworld.challenges import cooking
from textworld import EnvInfos

def test_TW():
    options = textworld.GameOptions()
    options.seeds = 5611

    settings = {
        'split': "test",
        'recipe_seed': 1234,
        'take': True,
        'open': True,
        'cook': True,
        'cut': True,
        'go': 6
        }
    game = cooking.make(settings, options)
    try:
        gamefile = compile_game(game)

        #request_infos = EnvInfos(description=True, inventory=True)
        request_infos = EnvInfos(description=True, inventory=True, entities=True, facts=True)
        env_id = textworld.gym.register_game(gamefile, request_infos)
        env = gym.make(env_id)
        ob, infos = env.reset()

        print(ob)
        print(infos)
    except:
        pass


def test_reset_step():
    gamefile = "./tests/tw_games/tw-cooking-test-take+cook+cut+open+go6-JrmLfNyMcErjF6LD.ulx"
    request_infos = EnvInfos(description=True, inventory=True, entities=True, facts=True)
    env_id = textworld.gym.register_game(gamefile, request_infos)
    env = gym.make(env_id)

    obs, infos = env.reset()

    print(obs)
    print(infos)
    assert type(obs) == str
    assert type(infos) == dict
    assert 'facts' in infos
    assert 'entities' in infos

    obs, reward, done, debug_info = env.step("go south")
    assert type(obs) == str
    assert type(reward) == int
    assert type(done) == bool
    assert type(debug_info) == dict
    assert 'facts' in debug_info
    assert 'entities' in debug_info
