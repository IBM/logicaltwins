import gym
import numpy as np
from collections import OrderedDict
import textworld, textworld.envs, textworld.logic
from textworld import EnvInfos
from textworld.envs import TextWorldEnv
from textworld.logic import GameLogic, Rule
import pddlgym, pddlgym.structs, pddlgym.parser
from pddlgym.structs import Literal, Predicate, Type, LiteralConjunction
from pddlgym.parser import Operator, PDDLDomain


def tw_signature_to_pddlgym(pred: textworld.logic.Signature) -> pddlgym.structs.Predicate:
    return Predicate(pred.name, len(pred.types), [Type(t) for t in pred.types])


def tw_predicate_to_pddlgym(pred: textworld.logic.Predicate) -> pddlgym.structs.Literal:
    assert type(pred.name) == str
    assert type(pred.names) == tuple # of str
    assert type(pred.types) == tuple # of str
    pddlgym_pred = Predicate(pred.name, len(pred.types), [Type(t) for t in pred.types])
    return pddlgym_pred(*pred.names) # pddlgym.Literal from pddlgym.Predicate


# Returns a list of "name:type", not a list of (name str, type str)
def parameters_from_tw_rule(rule: Rule) -> list:
    params = OrderedDict([i for pred in rule.preconditions for i in zip(pred.names, pred.types)]).items()
    return [n if t is None else f"{n}:{t}" for n, t in params]


# Operator:
#    self.name = name  # string
#    self.params = params  # list of structs.Type objects?  TypedEntity?
#              [(?a, A), (?b, B)]
#    self.preconds = preconds  # structs.Literal representing preconditions
#    self.effects = effects  # structs.Literal representing effects
def tw_rule_to_pddlgym(rule: Rule) -> pddlgym.parser.Operator:
    params = parameters_from_tw_rule(rule)
    # tuple of textworld.logic.Predicate -> (and ..)
    preconds = LiteralConjunction([tw_predicate_to_pddlgym(pred) for pred in rule.preconditions])
    # tuple of textworld.logic.Predicate -> (and ..)
    effects = LiteralConjunction([tw_predicate_to_pddlgym(pred) for pred in rule.postconditions])
    return Operator(name=rule.name, params=params, preconds=preconds, effects=effects)


def tw_gamelogic_to_pddlgym(gamelogic: GameLogic) -> pddlgym.parser.PDDLDomain:
    assert type(gamelogic.types) is textworld.logic.TypeHierarchy
    assert type(gamelogic.predicates) is set
    assert type(gamelogic.rules) is dict

    pddldomain = PDDLDomain(domain_name=None,
                        types=None,
                        type_hierarchy=None,
                        predicates=dict([(tw_pred.name, tw_signature_to_pddlgym(tw_pred)) for tw_pred in list(gamelogic.predicates)]),
                        operators=dict([(rule.name, tw_rule_to_pddlgym(rule)) for rule in gamelogic.rules.values()]))

    pddldomain.types = []
    pddldomain.actions = []
    pddldomain.constants = []

    return pddldomain


def pddlgym_domain_to_pddl(pddldomain: PDDLDomain) -> str:
    predicates = "\n\t".join([lit.pddl_str() for lit in pddldomain.predicates.values()])
    operators = "\n\t".join([op.pddl_str() for op in pddldomain.operators.values()])
    if pddldomain.constants:
        constants_str = "\n\t".join(list(sorted(map(lambda o: str(o).replace(":", " - "),
                                                pddldomain.constants))))
        constants = f"\n  (:constants {constants_str})\n"
    else:
        constants = ""
    requirements = ":typing"
    if "=" in pddldomain.predicates:
        requirements += " :equality"

    domain_str = """
(define (domain {})
(:requirements {})
(:types {})
{}
(:predicates {}
)
; (:actions {})
{}
{}
)
    """.format(pddldomain.domain_name, requirements, pddldomain._types_pddl_str(),
               constants, predicates, " ".join(map(str, pddldomain.actions)), operators,
               pddldomain._derived_preds_pddl_str())

    return domain_str


def tw_fact_to_pddlgym(fact: textworld.logic.Proposition) -> pddlgym.structs.Literal:
    return Predicate(fact.name, len(fact.names), [Type(t) for t in fact.types])(*fact.names)


def tw_facts_to_pddlgym(facts: list) -> frozenset:
    return frozenset([tw_fact_to_pddlgym(fact) for fact in facts])


def typebindings_from_tw_facts(facts: list) -> dict:
    # select
    types = frozenset([t for fact in facts for t in fact.types])
    pddlgym_types = dict(zip(types, [pddlgym.structs.Type(t) for t in list(types)]))
    return dict([i for fact in facts for i in zip(fact.names, [pddlgym_types[t] for t in fact.types])])


def tw_entities_to_pddlgym(entities: list, typebindings: dict) -> frozenset:
    from pddlgym.structs import NULLTYPE

    # apply pddlgym.Type instance as a function from name to typed name
    return frozenset([typebindings.get(entity, NULLTYPE)(entity) for entity in entities])


def tw_to_pddlgym_State(facts: list, entities: list, typebindings: dict) -> pddlgym.structs.State:
    return pddlgym.structs.State(
        literals = tw_facts_to_pddlgym(facts),
        objects = tw_entities_to_pddlgym(entities, typebindings),
        goal = None)

def tw_infos_to_pddlgym(infos: dict) -> pddlgym.structs.State:
    # we may have a list if batch_size specified
    facts = infos['facts'] # this can be list or list[list]
    entities = infos['entities'] # this can be list or list[list]

    if facts and type(facts[0]) is list:
        # batch mode
        typebindings = [typebindings_from_tw_facts(e) for e in facts]
        return [tw_to_pddlgym_State(e1, e2, e3) for e1, e2, e3 in zip(facts, entities, typebindings)]
    else:
        typebindings = typebindings_from_tw_facts(facts)
        return tw_to_pddlgym_State(facts, entities, typebindings)

# Example construction: Predicate('putdown', 1, [Type('block')])('a')
def pddlgym_action_to_tw(action: pddlgym.structs.Literal) -> str:
    """
    Available commands:
    look:                describe the current room
    goal:                print the goal of this game
    inventory:           print player's inventory
    go <dir>:            move the player north, east, south or west
    examine ...:         examine something more closely
    eat ...:             eat edible food
    open ...:            open a door or a container
    close ...:           close a door or a container
    drop ...:            drop an object on the floor
    take ...:            take an object that is on the floor
    put ... on ...:      place an object on a supporter
    take ... from ...:   take an object from a container or a supporter
    insert ... into ...: place an object into a container
    lock ... with ...:   lock a door or a container with a key
    unlock ... with ...: unlock a door or a container with a key
    """
    predicate = action.predicate.name # Predicate
    variables = action.variables # list of TypedEntity or str

    # we hard-code here
    if predicate == "close/c" or predicate == "close/d":
        # close/c :parameters (P - P r - r c - c)
        # close/d :parameters (P - P r - r d - d r' - r)
        return f"close {variables[2]}"
    elif predicate == "drop":
        # drop :parameters (P - P r - r o - o I - I)
        return f"drop {variables[2]}"
    elif predicate == "drink":
        # drink :parameters (f - f I - I slot - slot)
        return f"drink {variables[0]}"
    elif predicate == "eat":
        # eat :parameters (f - f I - I slot - slot)
        return f"eat {variables[0]}"
    elif predicate == "examine/t" or predicate == "examine/d":
        # examine/t :parameters (P - P r - r t - t)
        # examine/d :parameters (P - P r - r d - d r' - r)
        # examine t (in the current room) / d (to the room r')
        return f"examine {variables[2]}"
    elif predicate == "examine/I":
        # examine/I :parameters (o - o I - I)
        # examine o (in my inventory)
        return f"examine {variables[0]}"
    elif predicate == "examine/s" or predicate == "examine/c":
        # examine/s :parameters (P - P r - r s - s o - o)
        # examine/c :parameters (P - P r - r c - c o - o)
        # examine o (in c / on s)
        return f"examine {variables[3]}"
    elif predicate == "go/north":
        return "go north"
    elif predicate == "go/south":
        return "go south"
    elif predicate == "go/east":
        return "go east"
    elif predicate == "go/west":
        return "go west"
    elif predicate == "insert":
        # insert :parameters (P - P r - r c - c o - o I - I)
        return f"insert {variables[3]} into {variables[2]}"
    elif predicate == "inventory":
        return "inventory"
    elif predicate == "lock/c":
        # lock/c :parameters (P - P r - r c - c k - k I - I)
        return f"lock {variables[2]} with {variables[3]}"
    elif predicate == "lock/d":
        # lock/d :parameters (P - P r - r d - d r' - r k - k I - I)
        return f"lock {variables[2]} with {variables[4]}"
    elif predicate == "look":
        # look :parameters (P - P r - r)
        return "look"
    elif predicate == "open/c" or predicate == "open/d":
        # open/c :parameters (P - P r - r c - c)
        # open/d :parameters (P - P r - r d - d r' - r)
        return f"open {variables[2]}"
    elif predicate == "put":
        # put :parameters (P - P r - r s - s o - o I - I)
        return f"put {variables[3]} on {variables[2]}"
    elif predicate == "take":
        # take :parameters (P - P r - r o - o)
        return f"take {variables[2]}"
    elif predicate == "take/c":
        # take/c :parameters (P - P r - r c - c o - o)
        return f"take {variables[3]} from {variables[2]}"
    elif predicate == "take/s":
        # take/s :parameters (P - P r - r s - s o - o)
        return f"take {variables[3]} from {variables[2]}"
    elif predicate == "unlock/c":
        # unlock/c :parameters (P - P r - r c - c k - k I - I)
        return f"unlock {variables[2]} with {variables[3]}"
    elif predicate == "unlock/d":
        # unlock/d :parameters (P - P r - r d - d r' - r k - k I - I)
        return f"unlock {variables[2]} with {variables[4]}"
    elif predicate == "cook/oven/burned" or predicate == "cook/oven/cooked/raw" or predicate == "cook/oven/cooked/needs_cooking":
        # cook/oven/burned :parameters (P - P r - r oven - oven f - f I - I)
        # cook/oven/cooked/raw :parameters (P - P r - r oven - oven f - f I - I)
        # cook/oven/cooked/needs_cooking :parameters (P - P r - r oven - oven f - f I - I)
        return f"cook {variables[3]} in {variables[2]}"
    elif predicate == "cook/toaster/burned" or predicate == "cook/toaster/cooked/raw" or predicate == "cook/toaster/cooked/needs_cooking":
        # cook/toaster/burned :parameters (P - P r - r toaster - toaster f - f I - I)
        # cook/toaster/cooked/raw :parameters (P - P r - r toaster - toaster f - f I - I)
        # cook/toaster/cooked/needs_cooking :parameters (P - P r - r toaster - toaster f - f I - I)
        return f"cook {variables[3]} in {variables[2]}"
    elif predicate == "cook/stove/burned" or predicate == "cook/stove/cooked/raw" or predicate == "cook/stove/cooked/needs_cooking":
        # cook/stove/burned :parameters (P - P r - r stove - stove f - f I - I)
        # cook/stove/cooked/raw :parameters (P - P r - r stove - stove f - f I - I)
        # cook/stove/cooked/needs_cooking :parameters (P - P r - r stove - stove f - f I - I)
        return f"cook {variables[3]} on {variables[2]}"
    elif predicate == "slice" or predicate == "dice" or predicate == "chop":
        # slice :parameters (f - f I - I o - o)
        # dice :parameters (f - f I - I o - o)
        # chop :parameters (f - f I - I o - o)
        # slice/dice/chop f with o
        return f"{predicate} {variables[0]} with {variables[2]}"
    elif predicate == "make/recipe/1":
        # make/recipe/1 :parameters (P - P r - r RECIPE - RECIPE f - f I - I meal - meal)
        return f"make {variables[5]} from {variables[3]}"
    elif predicate == "make/recipe/2":
        # make/recipe/2 :parameters (P - P r - r RECIPE - RECIPE f - f I - I f' - f meal - meal slot - slot slot' - slot)
        return f"make {variables[6]} from {variables[3]} and {variables[5]}"
    elif predicate == "make/recipe/3":
        # make/recipe/3 :parameters (P - P r - r RECIPE - RECIPE f - f I - I f' - f f'' - f meal - meal slot - slot slot' - slot slot'' - slot)
        return f"make {variables[7]} from {variables[3]}, {variables[5]} and {variables[6]}"
    elif predicate == "make/recipe/4":
        # make/recipe/4 :parameters (P - P r - r RECIPE - RECIPE f - f I - I f' - f f'' - f f''' - f meal - meal slot - slot slot' - slot slot'' - slot)
        return f"make {variables[8]} from {variables[3]}, {variables[5]}, {variables[6]} and {variables[7]}"
    elif predicate == "make/recipe/5":
        # make/recipe/5 :parameters (P - P r - r RECIPE - RECIPE f - f I - I f' - f f'' - f f''' - f f'''' - f meal - meal slot - slot slot' - slot slot'' - slot slot''' - slot slot'''' - slot)
        return f"make {variables[9]} from {variables[3]}, {variables[5]}, {variables[6]}, {variables[7]} and {variables[8]}"
    else:
        # TODO better to throw errors to make sure we don't fall into here
        raise Exception(f"predicate {predicate} not supported yet")


class TextWorldWrapper(gym.Wrapper):
    """Turn a TextworldGym environment into one for PDDLGym.

    Parameters
    ----------
    env : TextworldEnv
    """

    metadata = {'render.modes': ['human', 'ansi', 'text']}

    def __init__(self, env: textworld.envs.TextWorldEnv):
        super().__init__(env)

        #self.action_space =
        #self.observation_space =

    def reset(self):
        self.text_obs, debug_info = super().reset()
        self.logical_obs = tw_infos_to_pddlgym(debug_info)
        return self.logical_obs, debug_info

    def step(self, logical_action: pddlgym.structs.Literal):
        text_action = pddlgym_action_to_tw(logical_action)
        self.text_obs, reward, done, debug_info = super().step(text_action)
        self.logical_obs = tw_infos_to_pddlgym(debug_info)
        return self.logical_obs, reward, done, debug_info

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): human/rgb_array/ansi the mode to render with
        """
        if mode == 'rgb_array':
            from .house import LogicalMultiRoomEnv
            # numpy array which can be visualized by matplotlib imshow()
            # https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html
            # (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            lenv = LogicalMultiRoomEnv(self.logical_obs)
            _obs = lenv.reset()
            return lenv.render(mode='rgb_array') # return RGB frame suitable for video
        elif mode == 'ansi':
            return self.text_obs
        elif mode == 'human':
            # pop up a window and render
            from IPython.core.display import display
            import pddlgym_textworld.rendering
            image = pddlgym_textworld.rendering.render(self.logical_obs)
            display(image)
        else:
            super(TextWorldWrapper, self).render(mode=mode) # just raise an exception
