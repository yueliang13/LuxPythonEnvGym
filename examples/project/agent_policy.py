import sys
import time
from functools import partial
import copy
import random

import numpy as np
from gymnasium import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)

def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        if target_unit is None:
                            target_unit = u
                        else:
                            if target_unit.type == u.type:
                                if( u.get_cargo_space_left() >= resource_amount and
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        target_unit = u
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    pass
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                target_unit = u

    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()

    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)


class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        super().__init__(mode, model)

        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),
            SpawnCityAction,
            PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        # Observation space:
        #   3x  object type (is_worker, is_cart, is_citytile)
        #   70x nearest + furthest for 5 object types (7 features each × 2)
        #   4x  unit cargo (wood, coal, uranium, space_left)   [NEW]
        #   1x  is night
        #   1x  turns until night                             [NEW]
        #   1x  percent of game done
        #   2x  citytile counts [cur player, opponent]
        #   2x  worker counts [cur player, opponent]
        #   2x  cart counts [cur player, opponent]
        #   1x  research points [cur player]
        #   1x  researched coal [cur player]
        #   1x  researched uranium [cur player]
        #   1x  opponent research points                      [NEW]
        #   1x  opponent citytile fuel ratio                  [NEW]
        self.observation_shape = (3 + 7 * 5 * 2 + 4 + 1 + 1 + 1 + 2 + 2 + 2 + 3 + 1 + 1,)
        self.observation_space = spaces.Box(low=0, high=1, shape=
        self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

    def get_agent_type(self):
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        obs = np.zeros(self.observation_shape)

        if is_new_turn:
            self.object_nodes = {}
            for cell in game.map.resources:
                if cell.resource.type not in self.object_nodes:
                    self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
                else:
                    self.object_nodes[cell.resource.type] = np.concatenate(
                        (self.object_nodes[cell.resource.type], [[cell.pos.x, cell.pos.y]]), axis=0
                    )

            for t in [team, (team + 1) % 2]:
                for u in game.state["teamStates"][t]["units"].values():
                    key = str(u.type)
                    if t != team:
                        key = str(u.type) + "_opponent"
                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (self.object_nodes[key], [[u.pos.x, u.pos.y]]), axis=0
                        )

            for city in game.cities.values():
                for cells in city.city_cells:
                    key = "city"
                    if city.team != team:
                        key = "city_opponent"
                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (self.object_nodes[key], [[cells.pos.x, cells.pos.y]]), axis=0
                        )

        # Object type
        observation_index = 0
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                obs[observation_index] = 1.0
            else:
                obs[observation_index + 1] = 1.0
        if city_tile is not None:
            obs[observation_index + 2] = 1.0
        observation_index += 3

        pos = None
        if unit is not None:
            pos = unit.pos
        else:
            pos = city_tile.pos

        if pos is None:
            observation_index += 7 * 5 * 2
        else:
            for distance_function in [closest_node, furthest_node]:
                for key in [
                    Constants.RESOURCE_TYPES.WOOD,
                    Constants.RESOURCE_TYPES.COAL,
                    Constants.RESOURCE_TYPES.URANIUM,
                    "city",
                    str(Constants.UNIT_TYPES.WORKER)]:
                    if key in self.object_nodes:
                        if (
                                (key == "city" and city_tile is not None) or
                                (unit is not None and str(unit.type) == key and len(game.map.get_cell_by_pos(unit.pos).units) <= 1)
                        ):
                            closest_index = closest_node((pos.x, pos.y), self.object_nodes[key])
                            filtered_nodes = np.delete(self.object_nodes[key], closest_index, axis=0)
                        else:
                            filtered_nodes = self.object_nodes[key]

                        if len(filtered_nodes) == 0:
                            obs[observation_index + 5] = 1.0
                        else:
                            closest_index = distance_function((pos.x, pos.y), filtered_nodes)
                            if closest_index is not None and closest_index >= 0:
                                closest = filtered_nodes[closest_index]
                                closest_position = Position(closest[0], closest[1])
                                direction = pos.direction_to(closest_position)
                                mapping = {
                                    Constants.DIRECTIONS.CENTER: 0,
                                    Constants.DIRECTIONS.NORTH: 1,
                                    Constants.DIRECTIONS.WEST: 2,
                                    Constants.DIRECTIONS.SOUTH: 3,
                                    Constants.DIRECTIONS.EAST: 4,
                                }
                                obs[observation_index + mapping[direction]] = 1.0
                                distance = pos.distance_to(closest_position)
                                obs[observation_index + 5] = min(distance / 20.0, 1.0)

                                if key == "city":
                                    c = game.cities[game.map.get_cell_by_pos(closest_position).city_tile.city_id]
                                    obs[observation_index + 6] = min(c.fuel / (c.get_light_upkeep() * 200.0), 1.0)
                                elif key in [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL,
                                             Constants.RESOURCE_TYPES.URANIUM]:
                                    obs[observation_index + 6] = min(
                                        game.map.get_cell_by_pos(closest_position).resource.amount / 500, 1.0
                                    )
                                else:
                                    obs[observation_index + 6] = min(
                                        next(iter(game.map.get_cell_by_pos(
                                            closest_position).units.values())).get_cargo_space_left() / 100, 1.0
                                    )

                    observation_index += 7

        # Unit cargo details [NEW]
        if unit is not None:
            obs[observation_index] = unit.cargo["WOOD"] / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            obs[observation_index + 1] = unit.cargo["COAL"] / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            obs[observation_index + 2] = unit.cargo["URANIUM"] / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            obs[observation_index + 3] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        observation_index += 4

        # Is night
        obs[observation_index] = game.is_night()
        observation_index += 1

        # Turns until night [NEW]
        turn = game.state["turn"]
        cycle_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        turn_in_cycle = turn % cycle_length
        day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        if turn_in_cycle < day_length:
            turns_until_night = (day_length - turn_in_cycle) / day_length
        else:
            turns_until_night = 0.0
        obs[observation_index] = turns_until_night
        observation_index += 1

        # Percent of game done
        obs[observation_index] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_index += 1

        # Citytile / worker / cart counts
        max_count = 30
        for key in ["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                obs[observation_index] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                obs[observation_index + 1] = len(self.object_nodes[(key + "_opponent")]) / max_count
            observation_index += 2

        # Research state
        obs[observation_index] = game.state["teamStates"][team]["researchPoints"] / 200.0
        obs[observation_index + 1] = float(game.state["teamStates"][team]["researched"]["coal"])
        obs[observation_index + 2] = float(game.state["teamStates"][team]["researched"]["uranium"])
        observation_index += 3

        # Opponent research points [NEW]
        opponent_team = (team + 1) % 2
        obs[observation_index] = game.state["teamStates"][opponent_team]["researchPoints"] / 200.0
        observation_index += 1

        # Opponent citytile fuel ratio [NEW]
        opponent_fuel = 0
        opponent_upkeep = 0
        for city in game.cities.values():
            if city.team == opponent_team:
                opponent_fuel += city.fuel
                opponent_upkeep += city.get_light_upkeep()
        if opponent_upkeep > 0:
            obs[observation_index] = min(opponent_fuel / (opponent_upkeep * 100.0), 1.0)
        observation_index += 1

        return obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile != None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game, unit_id=unit.id if unit else None, unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile, team=team, x=x, y=y
                )
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game, unit_id=unit.id if unit else None, unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile, team=team, x=x, y=y
                )
            return action
        except Exception as e:
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.research_points_last = 0
        self.researched_coal_last = False
        self.researched_uranium_last = False

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        if is_game_error:
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            return 0

        # Basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])
        city_tile_count = 0
        city_tile_count_opponent = 0
        my_fuel = 0
        opponent_fuel = 0
        for city in game.cities.values():
            if city.team == self.team:
                my_fuel += city.fuel
                for cell in city.city_cells:
                    city_tile_count += 1
            else:
                opponent_fuel += city.fuel
                for cell in city.city_cells:
                    city_tile_count_opponent += 1

        rewards = {}

        # 1. Unit creation/death reward (same as baseline)
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # 2. City tile creation/death reward (same as baseline)
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # 3. Fuel collected reward - weighted by resource value (IMPROVED)
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        fuel_delta = fuel_collected - self.fuel_collected_last
        rewards["rew/r_fuel_collected"] = fuel_delta / 20000
        self.fuel_collected_last = fuel_collected

        # 4. Research progress reward (NEW)
        research_points = game.state["teamStates"][self.team]["researchPoints"]
        research_delta = research_points - self.research_points_last
        rewards["rew/r_research"] = research_delta * 0.01

        # Bonus for unlocking new resources
        researched_coal = game.state["teamStates"][self.team]["researched"]["coal"]
        researched_uranium = game.state["teamStates"][self.team]["researched"]["uranium"]
        if researched_coal and not self.researched_coal_last:
            rewards["rew/r_unlock_coal"] = 1.0
        if researched_uranium and not self.researched_uranium_last:
            rewards["rew/r_unlock_uranium"] = 2.0
        self.research_points_last = research_points
        self.researched_coal_last = researched_coal
        self.researched_uranium_last = researched_uranium

        # 5. Night survival reward (NEW)
        if game.is_night():
            rewards["rew/r_night_survive"] = city_tile_count * 0.05

        # 6. Opponent suppression reward (NEW)
        rewards["rew/r_resource_control"] = (city_tile_count - city_tile_count_opponent) * 0.02

        # 7. Game end rewards
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            rewards["rew/r_city_tiles_end"] = city_tile_count
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 10.0
            else:
                rewards["rew/r_game_win"] = -10.0

        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward

    def turn_heurstics(self, game, is_first_turn):
        return