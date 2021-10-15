import gym
import textworld.gym
from textworld.generator import compile_game
from textworld.challenges import cooking
from textworld import EnvInfos


def test_rendering_build_dict_rooms():
    from pddlgym_textworld import tw_infos_to_pddlgym
    from pddlgym_textworld.rendering import build_dict_rooms

    gamefile = "./tests/tw_games/tw-cooking-test-take+cook+cut+open+go6-JrmLfNyMcErjF6LD.ulx"
    request_infos = EnvInfos(description=True, inventory=True, entities=True, facts=True)
    env_id = textworld.gym.register_game(gamefile, request_infos)
    env = gym.make(env_id)

    obs, infos = env.reset()
    pddlgym_state = tw_infos_to_pddlgym(infos)
    rooms_dict, first_room = build_dict_rooms(pddlgym_state)
    print(rooms_dict)

    # assert first_room == 'bathroom' # this changes every time
    assert rooms_dict == {
        'kitchen': [None, None, 'pantry', 'livingroom'],
        'livingroom': ['corridor', None, 'kitchen', None],
        'bedroom': [None, 'corridor', None, None],
        'corridor': ['bedroom', 'livingroom', 'bathroom', None],
        'pantry': [None, None, None, 'kitchen'],
        'bathroom': [None, None, None, 'corridor']
        }

    from pddlgym_textworld.rendering import render
    image = render(pddlgym_state)
    print(image)

def test_rendering_build_dict_game_goals():
    from pddlgym_textworld import tw_infos_to_pddlgym
    from pddlgym_textworld.rendering import build_dict_game_goals

    gamefile = "./tests/tw_games/tw-cooking-test-take+cook+cut+open+go6-JrmLfNyMcErjF6LD.ulx"
    request_infos = EnvInfos(description=True, inventory=True, entities=True, facts=True)
    env_id = textworld.gym.register_game(gamefile, request_infos)
    env = gym.make(env_id)

    obs, infos = env.reset()
    pddlgym_state = tw_infos_to_pddlgym(infos)
    game_goals = build_dict_game_goals(pddlgym_state)
    print(game_goals)
