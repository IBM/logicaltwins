import numpy as np
from gym_minigrid.minigrid import Grid, Goal, Wall, Door, COLOR_NAMES
from gym_minigrid.envs import MiniGridEnv
from gym_minigrid.envs.multiroom import Room
from .rendering import NSEW, build_dict_rooms, build_dict_game_goals


class Rooms:
    """
    zero origin to NW, following MiniGrid's coordinate system
    """

    def __init__(self, pddlgym_state, trim=True):
        self.pddlgym_state = pddlgym_state

        self.rooms_dict, _first_room = build_dict_rooms(pddlgym_state)
        self.rooms_xy = {}
        self.grid = np.full([self.num_rooms * 2 - 1, self.num_rooms * 2 - 1], None)

        # initialize with a random, first room
        self.placeRoom(self.names[0], *(self.num_rooms - 1, self.num_rooms - 1))

        if trim:
            self.trim()

        game_goals = build_dict_game_goals(pddlgym_state)
        self.location_at = game_goals['at']
        self.cooking_location = game_goals['cooking_location'] \
            if 'cooking_location' in game_goals else None
        self.secondary_goals = game_goals['secondary_goals'] \
            if 'secondary_goals' in game_goals else None

    def placeRoom(self, room, x, y):
        if not room in self.rooms_xy:
            self.rooms_xy[room] = [x, y]
            self.grid[x][y] = room

            north_room, south_room, east_room, west_room = self.rooms_dict[room]
            if north_room:
                self.placeRoom(north_room, x, y - 1)
            if south_room:
                self.placeRoom(south_room, x, y + 1)
            if east_room:
                self.placeRoom(east_room, x + 1, y)
            if west_room:
                self.placeRoom(west_room, x - 1, y)

    def trim(self):
        trim_min_x, trim_max_x = 0, self.grid.shape[0] - 1
        while not list(filter(lambda e: e, self.grid[trim_min_x])) and trim_min_x < trim_max_x:
            trim_min_x += 1
        while not list(filter(lambda e: e, self.grid[trim_max_x])) and trim_min_x < trim_max_x:
            trim_max_x -= 1

        trim_min_y, trim_max_y = 0, self.grid.shape[1] - 1
        while not list(filter(lambda e: e, self.grid[:,trim_min_y])) and trim_min_y < trim_max_y:
            trim_min_y += 1
        while not list(filter(lambda e: e, self.grid[:,trim_max_y])) and trim_min_y < trim_max_y:
            trim_max_y -= 1

        #print(((trim_min_x, trim_max_x), (trim_min_y, trim_max_y)))

        self.grid = self.grid[trim_min_x:trim_max_x + 1,trim_min_y:trim_max_y + 1]

        self.rooms_xy = dict([(name, [x - trim_min_x, y - trim_min_y]) for name, (x, y) in self.rooms_xy.items()])

    @property
    def names(self):
        return list(self.rooms_dict.keys())

    @property
    def num_rooms(self):
        return len(self.names)

    #@property
    #def agent_at(self):
    #    return self.location_at['P']


class LogicalMultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self, pddlgym_state, maxRoomSize=4, grid_size=25):
        assert maxRoomSize >= 4

        self.maxRoomSize = maxRoomSize

        self.logical_rooms = Rooms(pddlgym_state)

        self.numRooms = self.logical_rooms.num_rooms

        super(LogicalMultiRoomEnv, self).__init__(
            grid_size=grid_size,
            max_steps=self.numRooms * 20
        )

    def socketRoom(self, x, y, width, height):
        #print(f"socketRoom({x}, {y}, {width}, {height})")
        for i in range(x, x + width):
            for j in range(y, y + height):
                self.grid.set(i, j, None)

    def roomSize(self, room_name, room_widths, room_heights):
        X, Y = self.logical_rooms.rooms_xy[room_name]
        return room_widths[X], room_heights[Y]

    def roomLeftTop(self, room_name, room_widths, room_heights):
        X, Y = self.logical_rooms.rooms_xy[room_name]
        #print(f"{room_name} ({X}, {Y})")
        return 1 + sum(room_widths[:X]) + X, 1 + sum(room_heights[:Y]) + Y

    def roomCenter(self, room_name, room_widths, room_heights):
        X, Y = self.logical_rooms.rooms_xy[room_name]
        return (1 + sum(room_widths[:X]) + X) + room_widths[X] // 2, (1 + sum(room_heights[:Y]) + Y) + room_heights[Y] // 2

    def _gen_grid(self, width, height):
        # x, y grid?
        room_cols = self.logical_rooms.grid.shape[0]
        room_rows = self.logical_rooms.grid.shape[1]
        room_widths = [(width - 1) // room_cols - 1] * room_cols # e.g [3, 3, 3]
        room_heights = [(height - 1) // room_rows - 1] * room_rows # e.g [4, 4]

        #print((room_widths, room_heights))

        self.grid = Grid(width, height)

        wall = Wall()
        for i in range(0, width):
            for j in range(0, height):
                self.grid.set(i, j, wall)

        for room_name, (north, _south, _east, west) in self.logical_rooms.rooms_dict.items():
            room_lefttop = self.roomLeftTop(room_name, room_widths, room_heights)
            room_center = self.roomCenter(room_name, room_widths, room_heights)
            room_size = self.roomSize(room_name, room_widths, room_heights)

            self.socketRoom(*room_lefttop, *room_size)

            if north:
                door = Door(self._rand_elem(COLOR_NAMES)) # TODO: should be identifiable?
                self.grid.set(room_center[0], room_lefttop[1] - 1, door)
            if west:
                door = Door(self._rand_elem(COLOR_NAMES))
                self.grid.set(room_lefttop[0] - 1, room_center[1], door)

        #agent_at = self.logical_rooms.agent_at
        agent_at = self.logical_rooms.location_at['P']
        agent_room_lefttop = self.roomLeftTop(agent_at, room_widths, room_heights)
        agent_room_size = self.roomSize(agent_at, room_widths, room_heights)
        self.agent_dir = 2
        self.start_pos = self.place_agent(top=agent_room_lefttop, size=agent_room_size, rand_dir=False)

        # TODO: kitchen as a goal?
        # Place the final goal in the last room
        if self.logical_rooms.cooking_location:
            kitchen = self.logical_rooms.cooking_location
            kitchen_lefttop = self.roomLeftTop(kitchen, room_widths, room_heights)
            kitchen_size = self.roomSize(kitchen, room_widths, room_heights)
            self.goal_pos = self.place_obj(Goal(), top=kitchen_lefttop, size=kitchen_size)

        self.mission = 'traverse the rooms to get to the goal'

    def render(self, mode='rgb_array'):
        from PIL import Image, ImageDraw

        if mode == 'human':
            rgb_array = super().render(mode='rgb_array')
            image = Image.fromarray(rgb_array)
            draw = ImageDraw.Draw(image)
            display(image)
        else:
            return super().render(mode)
