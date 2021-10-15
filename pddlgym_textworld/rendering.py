from enum import IntEnum
from collections import namedtuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import torch

#from tw_cooking_game_puzzle.role_master_builder import write_clue


class NSEW(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


font = ImageFont.load_default()

def name_of(pddlgym_variable):
    from pddlgym.structs import TypedEntity

    return pddlgym_variable.name if type(pddlgym_variable) is TypedEntity else pddlgym_variable

def build_dict_rooms(pddlgym_state):
    '''
    creat a dictionary of rooms rooms_dict
    each room is a key the value is a table 1*4 corresponding to ['north','south','east','west']
    each column contain the name of a room or None
    :param infos:
    :return: rooms_dict
             first room: the room from which the size of the map will be calculated
    '''
    rooms_dict = dict()
    first_room = None  # room from which the map will be build
    ck_loc = None

    facts = pddlgym_state.literals
    for fact in facts:
        fact.name = str(fact.predicate)
        room0 = name_of(fact.variables[0])
        room1 = name_of(fact.variables[1]) if len(fact.variables) > 1 else None
        if fact.name == 'at' and room0 == 'P':
            first_room = room1
            rooms_dict[room1] = [None, None, None, None]
        elif fact.name in ['east_of', 'north_of']:
            if not first_room:
                first_room = room0
            if room0 not in rooms_dict:
                rooms_dict[room0] = [None, None, None, None]

            if room1 not in rooms_dict:
                rooms_dict[room1] = [None, None, None, None]

            if fact.name in 'east_of':
                rooms_dict[room0][NSEW.WEST] = room1
                rooms_dict[room1][NSEW.EAST] = room0
            else:
                rooms_dict[room0][NSEW.SOUTH] = room1
                rooms_dict[room1][NSEW.NORTH] = room0
        elif fact.name == 'cooking_location':
            ck_loc = room0

    if first_room == 'a':
        first_room = ck_loc if ck_loc else room0
        rooms_dict[first_room] = [None, None, None, None]
    return rooms_dict, first_room


def merge(dict1, dict2):
    '''

    :param dict1:
    :param dict2:
    :return: fusion of the two dictionaries (useful for build_dict_game_goals)
    '''
    res = {**dict1, **dict2}
    return res


def build_dict_game_goals(pddlgym_state):
    '''
    creat a dictionary containing the keys
    'at': associated to a dictionary where keys are objects or containers and values are the rooms where they are
    'base':associated to a table containing all the ingredients for the recipe
    'cooking_location': associated to the name of the place where cooking take place
    'secondary_goals':associated to a dictionary where keys are ingredients of the receipe and values are the rooms where they are

    :param infos:
    :return: dict_game_goals
    '''
    dict_game_goals = dict()
    dict_game_goals['base'] = []

    facts = pddlgym_state.literals
    for fact in sorted(facts, key=lambda f: str(f.predicate)):
        # sorted to start from at - a dirty quick hack
        fact.name = str(fact.predicate)
        room0 = name_of(fact.variables[0])
        room1 = name_of(fact.variables[1]) if len(fact.variables) > 1 else None
        if fact.name in ['at', 'in', 'on', 'base', 'cooking_location']:
            if fact.name == 'at':
                if fact.name not in dict_game_goals:
                    dict_game_goals[fact.name] = {room0: room1}
                else:
                    dict1 = dict_game_goals[fact.name]
                    dict2 = {room0: room1}
                    dict_game_goals[fact.name] = merge(dict1, dict2)
            elif fact.name == 'in' or fact.name == 'on':
                if room0 in dict_game_goals['base']:
                    if 'secondary_goals' not in dict_game_goals and 'I' != room1:
                        dict_game_goals['secondary_goals'] = {room0: dict_game_goals['at'][room1]}
                    elif 'secondary_goals' in dict_game_goals and 'I' != room1:
                        dict1 = dict_game_goals['secondary_goals']
                        dict2 = {room0: dict_game_goals['at'][room1]}
                        dict_game_goals['secondary_goals'] = merge(dict1, dict2)
            elif fact.name == 'base':
                dict_game_goals[fact.name].append(room0)
            else:
                dict_game_goals[fact.name] = room0
    return dict_game_goals


def picture_size(r_dict: dict, key: str, j: int, visited_rooms: dict, center_visited_rooms: dict):
    '''
    recursive function ( Depth-First Search) that explore all the rooms to deduce the minimum size of the final picture
    :param r_dict: the room dictionary see 'def build_dict_rooms'
    :param key: the room from which the algorithm explore
    :param j: an integer in [0;3] that indicates the position of the current room in regard to the previous room
              if j=0 that means that the current room is on the north of the previously visited room
    :param visited_rooms: a dictionary that registers if a room have already been visited
    :param center_visited_rooms: a dictionary that registers the center of the corresponding room
    :return: max_dir a 1*4 table of int that gives the max cordinates with first_room (0,0)
    :return: center_visited_rooms the center of all the rooms in the picture
    '''
    visited_rooms[key] = True

    direction = np.zeros((4, 4))
    end_path = True

    if j != -1:
        if j % 2 == 0:
            previous_room = r_dict[key][j + 1]
            if j == 0:
                center_visited_rooms[key] = center_visited_rooms[previous_room] + np.asarray([0, -100])
            else:
                center_visited_rooms[key] = center_visited_rooms[previous_room] + np.asarray([100, 0])
        else:
            previous_room = r_dict[key][j - 1]
            if j == 1:
                center_visited_rooms[key] = center_visited_rooms[previous_room] + np.asarray([0, +100])
            else:
                center_visited_rooms[key] = center_visited_rooms[previous_room] + np.asarray([-100, 0])

    for i in range(4):

        if r_dict[key][i] is not None and visited_rooms[r_dict[key][i]] is not True:

            mask = np.zeros(4)
            if i % 2 == 0:
                mask[i + 1] = -1
            else:
                mask[i - 1] = -1
            pic_size, dict_centers = picture_size(r_dict, r_dict[key][i], i, visited_rooms, center_visited_rooms)
            d = pic_size + mask
            d[d < 0] = 0
            direction[i] = d
            end_path = False

    if end_path:
        l = np.zeros(4)
        l[j] = 1
        return l, center_visited_rooms
    else:
        max_dir = np.max(direction, axis=0)

        if j != -1:
            max_dir[j] = max_dir[j] + 1
        else:
            for k in iter(r_dict):
                center_visited_rooms[k] = center_visited_rooms[k] + np.asarray(
                    [max_dir[3] * 100 + 50, max_dir[0] * 100 + 50])

        return max_dir, center_visited_rooms


def way_to_kitchen(rooms_dict: dict, key: str, visited_rooms: dict):
    visited_rooms[key] = True
    if key == 'kitchen':
        return ['kitchen']

    else:

        for i in range(4):
            if rooms_dict[key][i] is not None and visited_rooms[rooms_dict[key][i]] is not True:
                way = way_to_kitchen(rooms_dict, rooms_dict[key][i], visited_rooms)
                if way is not None:
                    return [key] + way


def pic_player_to_kitchen(rooms_dict: dict, dict_game_goals: dict, centers: dict, pic_size, room: str, room_name: bool,
                          dict_rooms_numbers: dict, name_type=['literal', 'random_numbers', 'room_importance'],
                          draw_passages=True, draw_player=True):
    visited_rooms = dict()
    for k in iter(rooms_dict):
        visited_rooms[k] = False

    way = way_to_kitchen(rooms_dict, room, visited_rooms)

    if len(centers) == 1:  # take care of the pathological case of a game with one room
        pic_size = np.zeros(4)
        for key in iter(centers):
            centers[key] = np.asarray([50, 50])
    # blue background image
    # im = Image.new('RGB',
    #                    (int((pic_size[3] + pic_size[2] + 1) * 100), int((pic_size[0] + pic_size[1] + 1) * 100)),
    #                    (100, 200, 255))

    # black background image
    im = Image.new('RGB',
                   (int((pic_size[3] + pic_size[2] + 1) * 100), int((pic_size[0] + pic_size[1] + 1) * 100)),
                   (0, 0, 0))
    draw = ImageDraw.Draw(im)

    for k in iter(centers):
        [x0, y0, x1, y1] = [centers[k][0] - 50, centers[k][1] - 50, centers[k][0] + 50, centers[k][1] + 50]
        if k in way:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), fill=(190, 245, 116))

        if draw_passages and k in way:
            dir_passages = [[centers[k][0] - 25, centers[k][1] - 50, centers[k][0] + 25, centers[k][1] - 45],
                            [centers[k][0] - 25, centers[k][1] + 45, centers[k][0] + 25, centers[k][1] + 50],
                            [centers[k][0] + 45, centers[k][1] - 25, centers[k][0] + 50, centers[k][1] + 25],
                            [centers[k][0] - 50, centers[k][1] - 25, centers[k][0] - 45, centers[k][1] + 25]]
            for i in range(4):
                if rooms_dict[k][i] is not None:
                    if k in way:
                        draw.rectangle(dir_passages[i], outline=(190, 245, 116), fill=(190, 245, 116))

            if draw_player and way and k == way[0]:
                draw.polygon([centers[k][0] + 50, centers[k][1] + 50, centers[k][0] + 35, centers[k][1] + 50,
                              centers[k][0] + 42, centers[k][1] + 20],
                             outline=(0, 0, 0), fill=(0, 0, 0))
                draw.ellipse([centers[k][0] + 37, centers[k][1] + 15, centers[k][0] + 47, centers[k][1] + 25],
                             outline=(0, 0, 0), fill=(0, 0, 0))
            if room_name:
                if name_type == 'literal' and k in way:
                    draw.text((centers[k][0] - 25, centers[k][1]), k, font=font, fill=(0, 0, 0),
                              stroke_width=0.2)
                elif name_type == 'room_importance' and k in way:
                    importance = 0
                    if k in dict_game_goals['cooking_location']:
                        importance = 1  # place of the main quest

                        draw.text((centers[k][0] - 15, centers[k][1]), "{}".format(importance), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)
                    if 'secondary_goals' in dict_game_goals and k in dict_game_goals['secondary_goals'].values():
                        importance = 2  # place of a secondary quest fo example to obtain an object

                        draw.text((centers[k][0] + 15, centers[k][1]), "{}".format(importance), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)
                    else:  # normal room
                        if k not in dict_game_goals['cooking_location']:
                            draw.text((centers[k][0], centers[k][1]), "{}".format(importance), font=font,
                                      fill=(0, 0, 0),
                                      stroke_width=0.2)
                else:
                    if k in way:
                        draw.text((centers[k][0], centers[k][1]), "{}".format(dict_rooms_numbers[k]), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)

    im.save("ev_cway.png")
    return im


def place_the_puzzle(rooms_dict: dict, key: str, visited_rooms: dict, random_place: bool, max_depth: int,
                     current_depth: int):
    '''
    recursive function (Deep-First Breath) that puts the hint a a distance max depth from the cooking place
    :param rooms_dict: dictionary of the rooms see 'def build_dict_rooms'
    :param key: the current place
    :param visited_rooms: the already visited rooms
    :param random_place: boolean if True the algorithm will browse the rooms randomly
    :param max_depth: the distance between the cooking place and the room with the hint
                      if this distance could not be reach the hint is place at the maximum distance from the cooking place
    :param current_depth: the current death of the exploration from the cooking place
    :return: when the algorithm terminates it retrieves a table l that gives the way between the cooking place and the
             room where is the hint and the finite distance len(l) between these two places len(l)<= max_depth
    '''
    visited_rooms[key] = True
    end_path = True

    depth = 0
    way_puzzle_room_to_cooking_place = []
    list_dir = list(range(0, 4))
    if random_place:  # browse the tree randomly to place the puzzle in different locations
        random.shuffle(list_dir)
    for i in list_dir:
        if max_depth > current_depth and rooms_dict[key][i] is not None \
                and visited_rooms[rooms_dict[key][i]] is not True:
            k, current_d = place_the_puzzle(rooms_dict, rooms_dict[key][i], visited_rooms, random_place, max_depth,
                                            current_depth + 1)
            if current_d > depth:
                depth = current_d
                way_puzzle_room_to_cooking_place = k

            end_path = False
    if end_path:
        return [key], 1
    else:
        l = [key] + way_puzzle_room_to_cooking_place
        return l, len(l)

def place_first_room(rooms_dict: dict, key: str, visited_rooms: dict, current_depth: int):

    visited_rooms[key] = True
    end_path = True


    if key == 'kitchen':
        return ['kitchen'], current_depth
    else:
        way_first_room_to_cooking_place = []
        for i in range(4):
            min_depth = len(rooms_dict)

            if rooms_dict[key][i] is not None and (visited_rooms[rooms_dict[key][i]] is not True or rooms_dict[key][i]=='kitchen'):
                k, depth = place_first_room(rooms_dict, rooms_dict[key][i], visited_rooms, current_depth+1)
                end_path = False

                if k is not None and 'kitchen' in k and depth < min_depth:

                    way_first_room_to_cooking_place = [key] + k

        if end_path:

            return None, current_depth
        else:
            return way_first_room_to_cooking_place, current_depth



def search_ways_to_secondary_goals(rooms_dict: dict, key: str, visited_rooms: dict, cooking_location: str):
    end_path = True
    rooms_part_of_the_way = None
    found_part_of_way = False

    if key != cooking_location:
        visited_rooms[key] = True

        for i in range(4):

            if rooms_dict[key][i] is not None and visited_rooms[rooms_dict[key][i]] is not True:
                k, found_part_of_way = search_ways_to_secondary_goals(rooms_dict, rooms_dict[key][i], visited_rooms,
                                                                      cooking_location)
                if found_part_of_way:
                    rooms_part_of_the_way = k
                end_path = False

    if end_path:
        if key == cooking_location:
            return set([key]), True
        else:
            return set(), False
    else:
        if rooms_part_of_the_way is not None:
            rooms_part_of_the_way.add(key)
            return rooms_part_of_the_way, True
        else:
            return set(), False


def rank_rooms(rooms_dict: dict, key: str, visited_rooms: dict, room_ranking: dict, useful_rooms: set):
    visited_rooms[key] = True
    if key in useful_rooms:
        room_ranking[key] = -1
    else:
        room_ranking[key] = 1

    for i in range(4):
        if rooms_dict[key][i] is not None and visited_rooms[rooms_dict[key][i]] is not True:
            val = rank_rooms(rooms_dict, rooms_dict[key][i], visited_rooms, room_ranking, useful_rooms)
            if val != -1 and room_ranking[key] != -1:
                room_ranking[key] = room_ranking[key] + val

    return room_ranking[key]


def place_death_room(rooms_dict: dict, dict_game_goals: dict, way: list, max_number_inaccessible_rooms: int):
    list_rooms_goals = [dict_game_goals['at']['P']]
    if 'secondary_goals' in dict_game_goals:
        for sgoal in iter(dict_game_goals['secondary_goals']):
            if dict_game_goals['secondary_goals'][sgoal] not in list_rooms_goals:
                list_rooms_goals.append(dict_game_goals['secondary_goals'][sgoal])
    # creat a set of all useful rooms (all rooms that are on the way to a primary or secondary goal)
    useful_rooms = set()
    for l in list_rooms_goals:
        visited_rooms = dict()
        for k in iter(rooms_dict):
            visited_rooms[k] = False
        s, b = (search_ways_to_secondary_goals(rooms_dict, l, visited_rooms, dict_game_goals['cooking_location']))
        useful_rooms = useful_rooms.union(s)

    useful_rooms = useful_rooms.union([dict_game_goals['cooking_location']])
    useful_rooms = useful_rooms.union(set(way))
    room_ranking = dict()
    visited_rooms = dict()
    for k in iter(rooms_dict):
        room_ranking[k] = 0
        visited_rooms[k] = False

    rank_rooms(rooms_dict, dict_game_goals['cooking_location'], visited_rooms, room_ranking, useful_rooms)
    reversed_room_ranking = {value: key for (key, value) in room_ranking.items()}
    death_room = None
    i = 1
    j = 0
    while i <= max_number_inaccessible_rooms:
        if i in reversed_room_ranking.keys():
            death_room = reversed_room_ranking[i]
            j = i
        i += 1

    return death_room, j


def draw_map(pic_size, centers: dict, rooms_dict: dict, dict_game_goals: dict, mask: bool, distance_of_puzzle: int,
             clue_first_room: bool, add_death_room: bool, max_number_inaccessible_rooms: int, room_name=True,
             color_way=True, name_type=['literal', 'random_numbers', 'room_importance'], draw_passages=True,
             draw_player=True, random_place=False, name="map"):
    '''

    :param pic_size: size of the final picture with the center of coordinates first_room is (0,0) (see def picture_size)
    :param centers: the centers of all the rooms
    :param rooms_dict: dictionary of the rooms see 'def build_dict_rooms'
    :param dict_game_goals: a dictionary with the main goals and place of important elements for the game see( def build_dict_game_goals)
    :param dict_game_goals: if true only see the color way
    :param distance_of_puzzle: the distance between the cooking place and the room with the hint
    :param add_death_room: if you want to add a death room
    :param max_number_inaccessible_rooms: number of rooms that will be blocked by the death room
    :param room_name: boolean if True the room will be named
    :param color_way: boolean if True the room in the way between the cooking place and the hint's room
    :param name_type: three possibilities 'literal' : the true name of the room (eg. kitchen, bedroom)
                                          'random_numbers': a random number is attributed to each room
                                          'room_importance': each room receive a number based on the importance of the room
                                                             1 is for a room where which is a cooking place
                                                             2 is for a room where there is an ingredient for the recipe
                                                             0 if the room is uninteresting
    :param draw_passages: boolean if true draw the passage between the rooms
    :param draw_player: boolean if true draw the passage between the player on the map
    :param random_place: boolean if True the algorithm will browse the rooms randomly see( def place_the_puzzle)
    :param name: the name of the jpeg file containing the map and the txt file containing the game master information
    :return: way: a table that gives the way between the cooking place and the room where is the hint
             np.array(im): an array representing the map
    '''

    if len(centers) == 1:  # take care of the pathological case of a game with one room
        pic_size = np.zeros(4)
        for key in iter(centers):
            centers[key] = np.asarray([50, 50])

    # black background image
    im = Image.new('RGB',
                   (int((pic_size[3] + pic_size[2] + 1) * 100), int((pic_size[0] + pic_size[1] + 1) * 100)),
                   (0, 0, 0))
    draw = ImageDraw.Draw(im)

    visited_rooms = dict()

    for k in iter(rooms_dict):
        visited_rooms[k] = False

    if clue_first_room:
        way, d = place_first_room(rooms_dict, dict_game_goals['at']['P'], visited_rooms, 1)
        way = [x for x in reversed(way)]
    else:
        way, d = place_the_puzzle(rooms_dict, dict_game_goals['cooking_location'], visited_rooms, random_place,
                                  distance_of_puzzle, 1)

    if add_death_room:
        death_room, number_inaccessible_rooms = place_death_room(rooms_dict, dict_game_goals, way,
                                                                 max_number_inaccessible_rooms=max_number_inaccessible_rooms)

    else:
        death_room = None
        number_inaccessible_rooms = 0

    if add_death_room and death_room is None:
        print('there is no possibility to add a death room')

    if name_type == 'random_numbers':
        list_name = list(range(0, len(centers)))
        random.shuffle(list_name)

    dict_rooms_numbers = dict()  # a dictionary where key are rooms and values the numbers by which these rooms are designated

    iter_name = 0
    if mask and color_way:
        for k in iter(centers):

            [x0, y0, x1, y1] = [centers[k][0] - 50, centers[k][1] - 50, centers[k][0] + 50, centers[k][1] + 50]
            if color_way and k in way:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), fill=(190, 245, 116))
            if draw_passages and k in way:
                dir_passages = [[centers[k][0] - 25, centers[k][1] - 50, centers[k][0] + 25, centers[k][1] - 45],
                                [centers[k][0] - 25, centers[k][1] + 45, centers[k][0] + 25, centers[k][1] + 50],
                                [centers[k][0] + 45, centers[k][1] - 25, centers[k][0] + 50, centers[k][1] + 25],
                                [centers[k][0] - 50, centers[k][1] - 25, centers[k][0] - 45, centers[k][1] + 25]]
                for i in range(4):
                    if rooms_dict[k][i] is not None:
                        if color_way and k in way:
                            draw.rectangle(dir_passages[i], outline=(190, 245, 116), fill=(190, 245, 116))
                        elif death_room is not None and k == death_room:
                            draw.rectangle(dir_passages[i], outline=(200, 8, 21), fill=(200, 8, 21))
                        else:
                            draw.rectangle(dir_passages[i], outline=(255, 255, 255), fill=(255, 255, 255))
            if draw_player and way and k == way[-1]:
                draw.polygon([centers[k][0] + 50, centers[k][1] + 50, centers[k][0] + 35, centers[k][1] + 50,
                              centers[k][0] + 42, centers[k][1] + 20],
                             outline=(0, 0, 0), fill=(0, 0, 0))
                draw.ellipse([centers[k][0] + 37, centers[k][1] + 15, centers[k][0] + 47, centers[k][1] + 25],
                             outline=(0, 0, 0), fill=(0, 0, 0))
            if room_name:
                if name_type == 'literal':
                    draw.text((centers[k][0] - 25, centers[k][1]), k, font=font, fill=(0, 0, 0),
                              stroke_width=0.2)
                elif name_type == 'room_importance':
                    importance = 0
                    if k in dict_game_goals['cooking_location']:
                        importance = 1  # place of the main quest

                        draw.text((centers[k][0] - 15, centers[k][1]), "{}".format(importance), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)
                    if 'secondary_goals' in dict_game_goals and k in dict_game_goals['secondary_goals'].values():
                        importance = 2  # place of a secondary quest fo example to obtain an object

                        draw.text((centers[k][0] + 15, centers[k][1]), "{}".format(importance), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)
                    else:  # normal room
                        if k not in dict_game_goals['cooking_location']:
                            draw.text((centers[k][0], centers[k][1]), "{}".format(importance), font=font,
                                      fill=(0, 0, 0),
                                      stroke_width=0.2)

                else:
                    dict_rooms_numbers[k] = list_name[iter_name]
                    draw.text((centers[k][0], centers[k][1]), "{}".format(list_name[iter_name]), font=font,
                              fill=(0, 0, 0),
                              stroke_width=0.2)
            iter_name += 1
    else:
        for k in iter(centers):

            [x0, y0, x1, y1] = [centers[k][0] - 50, centers[k][1] - 50, centers[k][0] + 50, centers[k][1] + 50]
            if color_way and k in way:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), fill=(190, 245, 116))
            elif death_room is not None and k == death_room:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), fill=(200, 8, 21))
            else:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), fill=(255, 255, 255))

            if draw_passages:
                dir_passages = [[centers[k][0] - 25, centers[k][1] - 50, centers[k][0] + 25, centers[k][1] - 45],
                                [centers[k][0] - 25, centers[k][1] + 45, centers[k][0] + 25, centers[k][1] + 50],
                                [centers[k][0] + 45, centers[k][1] - 25, centers[k][0] + 50, centers[k][1] + 25],
                                [centers[k][0] - 50, centers[k][1] - 25, centers[k][0] - 45, centers[k][1] + 25]]
                for i in range(4):
                    if rooms_dict[k][i] is not None:
                        if color_way and k in way:
                            draw.rectangle(dir_passages[i], outline=(190, 245, 116), fill=(190, 245, 116))
                        elif death_room is not None and k == death_room:
                            draw.rectangle(dir_passages[i], outline=(200, 8, 21), fill=(200, 8, 21))
                        else:
                            draw.rectangle(dir_passages[i], outline=(255, 255, 255), fill=(255, 255, 255))
            if draw_player and k == way[-1]:
                draw.polygon([centers[k][0] + 50, centers[k][1] + 50, centers[k][0] + 35, centers[k][1] + 50,
                              centers[k][0] + 42, centers[k][1] + 20],
                             outline=(0, 0, 0), fill=(0, 0, 0))
                draw.ellipse([centers[k][0] + 37, centers[k][1] + 15, centers[k][0] + 47, centers[k][1] + 25],
                             outline=(0, 0, 0), fill=(0, 0, 0))
            if room_name:
                if name_type == 'literal':
                    draw.text((centers[k][0] - 25, centers[k][1]), k, font=font, fill=(0, 0, 0),
                              stroke_width=0.2)
                elif name_type == 'room_importance':
                    importance = 0
                    if k in dict_game_goals['cooking_location']:
                        importance = 1  # place of the main quest

                        draw.text((centers[k][0] - 15, centers[k][1]), "{}".format(importance), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)
                    if 'secondary_goals' in dict_game_goals and k in dict_game_goals['secondary_goals'].values():
                        importance = 2  # place of a secondary quest fo example to obtain an object

                        draw.text((centers[k][0] + 15, centers[k][1]), "{}".format(importance), font=font,
                                  fill=(0, 0, 0),
                                  stroke_width=0.2)
                    else:  # normal room
                        if k not in dict_game_goals['cooking_location']:
                            draw.text((centers[k][0], centers[k][1]), "{}".format(importance), font=font,
                                      fill=(0, 0, 0),
                                      stroke_width=0.2)

                else:
                    dict_rooms_numbers[k] = list_name[iter_name]
                    draw.text((centers[k][0], centers[k][1]), "{}".format(list_name[iter_name]), font=font,
                              fill=(0, 0, 0),
                              stroke_width=0.2)
            iter_name += 1
    # write to stdout
    im.save(name + "_map.png")
    #rooms_leading_to_death_room = write_clue(name + "_role_master.txt", way, rooms_dict, dict_game_goals,
    #                                         dict_rooms_numbers, death_room=death_room,
    #                                         name_type=name_type)
    rooms_leading_to_death_room = None
    return way, death_room, rooms_leading_to_death_room, dict_rooms_numbers, number_inaccessible_rooms, im  # np.array(im)


class Rendering:

    def __init__(self, batch_size: int,
                 mask: bool, distance_of_puzzle: int, clue_first_room: bool, add_death_room: bool,
                 max_number_inaccessible_rooms: int, room_name=True, color_way=True, upgradable_color_way=True,
                 name_type=["literal", 'random_numbers', 'room_importance'], draw_passages=True, draw_player=True,
                 level_clue=['easy', 'medium', 'hard', 'very hard'], random_place=False, name="map"):
        """
        Create Role Master
        The Role Master is an intermediate between the environment and the agent
        He can add extra information and display some puzzles

        Agent <-> Role Master <-> environment
        """

        self.infos = {}

        self.batch_size = batch_size
        self.rooms_dict = dict()
        self.mask = mask  # If you want to mask all  rooms excepted the ones in the colorway
        self.distance_of_puzzle = distance_of_puzzle  # if clue_first_room room distance of the puzzle is no considered
        self.clue_first_room = clue_first_room
        self.add_death_room = add_death_room
        self.death_room = None
        self.rooms_leading_to_death_room = dict()
        self.beginning_room = None
        self.max_number_inaccessible_rooms = max_number_inaccessible_rooms
        self.room_name = room_name
        self.color_way = color_way
        self.upgradable_color_way = upgradable_color_way
        self.name_type = name_type
        self.draw_passages = draw_passages
        self.draw_player = draw_player
        self.level_clue = level_clue
        self.random_place = random_place
        self.name = name

        self.im = torch.zeros(3, 500, 500, dtype=torch.float)  # the picture in the hint
        self.no_im = torch.zeros(3, 500, 500, dtype=torch.float)  # the picture in the hint

        self.hint = ''  # the hint to understand the picture by the agent
        self.indication_deathroom = ''  # the clue to avoid the death_room by the agent
        self.take_hint = [False] * batch_size  # the memory that the agent has found and has taken the hint
        self.read_hint = [False] * batch_size  # the memory of the hint to understand the picture by the agent
        self.read_indication_deathroom = [False] * batch_size  # the memory of the clue to avoid the death_room by the
        # agent

        self.current_room = ['']*batch_size
        self.room_of_the_hint = -1
        self.level_clue = level_clue

        self.rewards_hint = [1] * self.batch_size
        self.rewards_board = [1] * self.batch_size
        self.rewards_death = [-1] * self.batch_size

    def render(self, infos):
        self.rooms_dict, first_room = build_dict_rooms(infos)
        self.dict_game_goals = build_dict_game_goals(infos)
        self.beginning_room = self.dict_game_goals['at']['P']
        self.current_room = [self.beginning_room]*self.batch_size

        # size of the picture
        visited_rooms = dict()
        self.center_visited_rooms = dict()

        for k in iter(self.rooms_dict):
            visited_rooms[k] = False
            self.center_visited_rooms[k] = np.zeros(2)

        self.pic_size, self.center_visited_rooms = picture_size(self.rooms_dict, first_room, -1, visited_rooms,
                                                                self.center_visited_rooms)

        # draw the map
        _way, _death_room, _rooms_leading_to_death_room, _dict_rooms_nbr, _, self.im = draw_map(
            self.pic_size,
            self.center_visited_rooms,
            self.rooms_dict,
            self.dict_game_goals,
            mask=self.mask,
            distance_of_puzzle=self.distance_of_puzzle,
            clue_first_room=self.clue_first_room,
            add_death_room=self.add_death_room,
            max_number_inaccessible_rooms=self.max_number_inaccessible_rooms,
            room_name=self.room_name,
            color_way=self.room_name,
            name_type=self.name_type,
            draw_passages=self.draw_passages,
            draw_player=self.draw_player,
            random_place=self.random_place,
            name=self.name)

        return self.im


def render(infos, mode='human', close=False):
    params = Rendering(batch_size=1, mask=False,
                          distance_of_puzzle=4, add_death_room=False, clue_first_room=True, max_number_inaccessible_rooms=2,
                          room_name=True, color_way=True, upgradable_color_way=True, name_type='literal', draw_passages=True,
                          draw_player=True, level_clue='easy', random_place=True, name='cooking_game')

    params.rooms_dict, first_room = build_dict_rooms(infos)
    params.dict_game_goals = build_dict_game_goals(infos)
    params.beginning_room = params.dict_game_goals['at']['P']
    params.current_room = [params.beginning_room]*params.batch_size

    # size of the picture
    visited_rooms = dict()
    params.center_visited_rooms = dict()

    for k in iter(params.rooms_dict):
        visited_rooms[k] = False
        params.center_visited_rooms[k] = np.zeros(2)

    params.pic_size, params.center_visited_rooms = picture_size(params.rooms_dict, first_room, -1, visited_rooms,
                                                            params.center_visited_rooms)

    _way, _death_room, _rooms_leading_to_death_room, _dict_rooms_nbr, _, im = draw_map(
        params.pic_size,
        params.center_visited_rooms,
        params.rooms_dict,
        params.dict_game_goals,
        mask=params.mask,
        distance_of_puzzle=params.distance_of_puzzle,
        clue_first_room=params.clue_first_room,
        add_death_room=params.add_death_room,
        max_number_inaccessible_rooms=params.max_number_inaccessible_rooms,
        room_name=params.room_name,
        color_way=params.room_name,
        name_type=params.name_type,
        draw_passages=params.draw_passages,
        draw_player=params.draw_player,
        random_place=params.random_place,
        name=params.name)

    return im
