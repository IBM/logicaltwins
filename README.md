# TextWorld-to-PDDLGym

This is a simple library for converting TextWorld environment (an OpenAI gym) to PDDLGym's.
* TextWorld https://github.com/microsoft/TextWorld
* PDDLGym https://github.com/tomsilver/pddlgym

For installation

    !pip install .

to go.  It depends on (and automatically installs) TextWorld and PDDLGym.

First, you may generate your own TextWorld game.

    import textworld
    from textworld.generator import make_game, compile_game
    from textworld.challenges import cooking

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
    gamefile = compile_game(game)

Or, your can use a given game file.

    gamefile = "./demo_games/tw-cooking-test-take+cook+cut+open+go6-JrmLfNyMcErjF6LD.ulx"

Then, you should provide a TextWorld enviornment, specifying some EnvInfos options as follows.

    import gym
    import textworld.gym
    from textworld import EnvInfos

    request_infos = EnvInfos(description=True, entities=True, facts=True)
    env_id = textworld.gym.register_game(gamefile, request_infos)
    env = gym.make(env_id)
    obs, infos = env.reset()

Now you are ready to go with TextWorld.

    from pddlgym_textworld import TextWorldWrapper
    
    env = TextWorldWrapper(tw_env)

    ob, infos = env.reset()

Observation (ob) is in pddlgym.structs.State.  You can consider the wrappeed environment as PDDLGym's logical environment.

    from pddlgym.structs import Literal, Predicate, Type
    
    #obs, reward, done, debug_info = env.step("go south")
    action = Predicate('go/south', 3, [Type('P'), Type('r'), Type('r')])('P', 'bedroom', 'corridor')
    obs, reward, done, debug_info = env.step(action)
