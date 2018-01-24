import heapq
import numpy as np
import os
import time

import hlt
from cnnbot.common import *
from cnnbot.neural_net import ConvNeuralNet
from cnnbot.map_transform import resize_frame

class Bot:
    def __init__(self, location, name):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_location = os.path.join(current_directory, os.path.pardir, "models", location)
        self._name = name
        self._neural_net = ConvNeuralNet(cached_model=model_location)

        # Run prediction on random data to make sure that code path is executed at least once before the game starts
        random_input_data = np.random.rand(64,64,3)
        predictions = self._neural_net.predict(random_input_data)
        assert len(predictions) == PLANET_MAX_NUM

    def play(self):
        """
        Play a game using stdin/stdout.
        """

        # Initialize the game.
        game = hlt.Game(self._name)

        while True:
            # Update the game map.
            game_map = game.update_map()
            start_time = time.time()

            # Produce features for each planet.
            features = self.produce_features(game_map)

            # Find predictions which planets we should send ships to.
            predictions = self._neural_net.predict(features)

            # Use simple greedy algorithm to assign closest ships to each planet according to predictions.
            ships_to_planets_assignment = self.produce_ships_to_planets_assignment(game_map, predictions)

            # Produce halite instruction for each ship.
            instructions = self.produce_instructions(game_map, ships_to_planets_assignment, start_time)

            # Send the command.
            game.send_command_queue(instructions)

    def produce_features(self, game_map):
        new_array = np.zeros((game_map.width, game_map.height,3))

        for planet in game_map.all_planets():
            planet_id = planet.id
            planet_x = int(planet.x)
            planet_y = int(planet.y)
            planet_r = planet.radius
            health = planet.health
            production = planet.current_production
            for x in range(2*int(planet_r)+1):
                for y in range(2*int(planet_r)+1):
                    new_array[(planet_x+int(planet_r) -x, planet_y+int(planet_r) - y, 0)] = health/(int(planet_r)**2)
                    if planet.owner == game_map.get_me():
                        new_array[(planet_x+int(planet_r) -x,planet_y+int(planet_r) - y, 1)] = 1
                    elif planet.owner  == None:
                        new_array[(planet_x+int(planet_r) -x,planet_y+int(planet_r) - y, 1)] = 0
                    else:
                        new_array[(planet_x+int(planet_r) -x,planet_y+int(planet_r) - y, 1)] = -1
                    new_array[(planet_x+int(planet_r) -x,planet_y+int(planet_r) - y, 2)] = production
                    y += 1
                x += 1
                y = y + 2*int(planet_r)+1
        for player in game_map.all_players():
            for ship in player.all_ships():
                x = int(ship.x)
                y = int(ship.y)
                new_array[(x,y,0)] = new_array[(x,y,0)] + ship.health
                if ship.owner == game_map.get_me():
                    new_array[(x,y,1)] = 1
                else:
                    new_array[(x,y,1)] = -1
        return(resize_frame(new_array))

    def produce_ships_to_planets_assignment(self, game_map, predictions):
        """
        Given the predictions from the neural net, create assignment (undocked ship -> planet) deciding which
        planet each ship should go to. Note that we already know how many ships is going to each planet
        (from the neural net), we just don't know which ones.

        :param game_map: game map
        :param predictions: probability distribution describing where the ships should be sent
        :return: list of pairs (ship, planet)
        """
        undocked_ships = [ship for ship in game_map.get_me().all_ships()
                          if ship.docking_status == ship.DockingStatus.UNDOCKED]

        # greedy assignment
        assignment = []
        number_of_ships_to_assign = len(undocked_ships)

        if number_of_ships_to_assign == 0:
            return []

        planet_heap = []
        ship_heaps = [[] for _ in range(PLANET_MAX_NUM)]

        # Create heaps for greedy ship assignment.
        for planet in game_map.all_planets():
            # We insert negative number of ships as a key, since we want max heap here.
            heapq.heappush(planet_heap, (-predictions[planet.id] * number_of_ships_to_assign, planet.id))
            h = []
            for ship in undocked_ships:
                d = ship.calculate_distance_between(planet)
                heapq.heappush(h, (d, ship.id))
            ship_heaps[planet.id] = h

        # Create greedy assignment
        already_assigned_ships = set()

        while number_of_ships_to_assign > len(already_assigned_ships):
            # Remove the best planet from the heap and put it back in with adjustment.
            # (Account for the fact the distribution values are stored as negative numbers on the heap.)
            ships_to_send, best_planet_id = heapq.heappop(planet_heap)
            ships_to_send = -(-ships_to_send - 1)
            heapq.heappush(planet_heap, (ships_to_send, best_planet_id))

            # Find the closest unused ship to the best planet.
            _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])
            while best_ship_id in already_assigned_ships:
                _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])

            # Assign the best ship to the best planet.
            assignment.append(
                (game_map.get_me().get_ship(best_ship_id), game_map.get_planet(best_planet_id)))
            already_assigned_ships.add(best_ship_id)

        return assignment

    def produce_instructions(self, game_map, ships_to_planets_assignment, round_start_time):
        """
        Given list of pairs (ship, planet) produce instructions for every ship to go to its respective planet.
        If the planet belongs to the enemy, we go to the weakest docked ship.
        If it's ours or is unoccupied, we try to dock.

        :param game_map: game map
        :param ships_to_planets_assignment: list of tuples (ship, planet)
        :param round_start_time: time (in seconds) between the Epoch and the start of this round
        :return: list of instructions to send to the Halite engine
        """
        command_queue = []
        # Send each ship to its planet
        for ship, planet in ships_to_planets_assignment:
            speed = hlt.constants.MAX_SPEED

            is_planet_friendly = not planet.is_owned() or planet.owner == game_map.get_me()

            if is_planet_friendly:
                if ship.can_dock(planet):
                    command_queue.append(ship.dock(planet))
                else:
                    command_queue.append(
                        self.navigate(game_map, round_start_time, ship, ship.closest_point_to(planet), speed))
            else:
                docked_ships = planet.all_docked_ships()
                assert len(docked_ships) > 0
                weakest_ship = None
                for s in docked_ships:
                    if weakest_ship is None or weakest_ship.health > s.health:
                        weakest_ship = s
                command_queue.append(
                    self.navigate(game_map, round_start_time, ship, ship.closest_point_to(weakest_ship), speed))
        return command_queue

    def navigate(self, game_map, start_of_round, ship, destination, speed):
        """
        Send a ship to its destination. Because "navigate" method in Halite API is expensive, we use that method only if
        we haven't used too much time yet.

        :param game_map: game map
        :param start_of_round: time (in seconds) between the Epoch and the start of this round
        :param ship: ship we want to send
        :param destination: destination to which we want to send the ship to
        :param speed: speed with which we would like to send the ship to its destination
        :return:
        """
        current_time = time.time()
        have_time = current_time - start_of_round < 1.2
        navigate_command = None
        if have_time:
            navigate_command = ship.navigate(destination, game_map, speed=speed, max_corrections=180)
        if navigate_command is None:
            # ship.navigate may return None if it cannot find a path. In such a case we just thrust.
            dist = ship.calculate_distance_between(destination)
            speed = speed if (dist >= speed) else dist
            navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
        return navigate_command
