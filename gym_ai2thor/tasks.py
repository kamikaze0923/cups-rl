"""
Different task implementations that can be defined inside an ai2thor environment
"""

from gym_ai2thor.utils import InvalidTaskParams


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config.get('max_episode_length', 1000)
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class PickUpTask(BaseTask):
    """
    This task consists of picking up a target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details). Because the agent can only carry 1 object at a time in its inventory, to receive
    a lot of reward one must learn to put objects down. Optimal behaviour will lead to the agent
    spamming PickupObject and PutObject near a receptacle. target_objects is a dict which contains
    the target objects which the agent gets reward for picking up and the amount of reward was the
    value
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # check that target objects are not selected as NON pickupables
        missing_objects = []
        for obj in kwargs['task']['target_objects'].keys():
            if obj not in kwargs['pickup_objects']:
                missing_objects.append(obj)
        if missing_objects:
            raise InvalidTaskParams('Error initializing PickUpTask. The objects {} are not '
                                    'pickupable!'.format(missing_objects))

        self.target_objects = kwargs['task'].get('target_objects', {})
        self.prev_inventory = []

    def transition_reward(self, state, action_str=None):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] in self.target_objects

        if object_picked_up:
            # One of the Target objects has been picked up. Add reward from the specific object
            reward += self.target_objects.get(curr_inventory[0]['objectType'], 0)
            print('{} reward collected!'.format(reward))

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            # print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        self.prev_inventory = []
        self.step_num = 0

class PickUpAndFindReceptacleTask(BaseTask):
    """
    This task consists of picking up a target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details). Because the agent can only carry 1 object at a time in its inventory, to receive
    a lot of reward one must learn to put objects down. Optimal behaviour will lead to the agent
    spamming PickupObject and PutObject near a receptacle. Hear we specify a target receptacle for
    maximum reward to see if the agent would be able to learn this. target_objects is a dict which
    contains the target objects which the agent gets reward for picking up and the amount of reward
    was the value
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # check that target objects are not selected as NON pickupables
        missing_objects = []
        for obj in kwargs['task']['target_objects'].keys():
            if obj not in kwargs['pickup_objects']:
                missing_objects.append(obj)
        if missing_objects:
            raise InvalidTaskParams('Error initializing PickUpTask. The objects {} are not '
                                    'pickupable!'.format(missing_objects))

        self.target_objects = kwargs['task'].get('target_objects', {})
        self.target_receptacles = kwargs['task'].get('target_receptacles', {})
        self.target_receptacles_need_open = kwargs['task'].get('target_receptacles_need_open', {})
        self.prev_inventory = []

    def transition_reward(self, state, action_str=None):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] in self.target_objects

        object_put_down = self.prev_inventory and not curr_inventory and \
                           self.prev_inventory[0]['objectType'] in self.target_objects

        if object_picked_up:
            # One of the Target objects has been picked up. Add reward from the specific object
            special_reward = self.target_objects.get(curr_inventory[0]['objectType'], 0)
            # print('Pick up {}, {} reward collected!'.format(curr_inventory[0]['objectType'], special_reward))
            reward += special_reward

        if object_put_down:
            # One of the Target objects has been put down. Check the receptacle and add reward from the specific
            # receptacle
            receptacle = state.metadata["lastObjectPutReceptacle"]['objectType']
            special_reward = self.target_receptacles.get(receptacle, 0)
            # print('Put down to {}, {} reward collected!'.format(receptacle, special_reward))
            reward += special_reward


        if action_str == "OpenObject" and state.metadata["lastObjectOpened"] is not None:
            special_reward = self.target_receptacles_need_open.get(state.metadata["lastObjectOpened"]['objectType'], 0)
            # print('Opened {}, {} reward collected!'.format(opened_object['objectType'], special_reward))
            reward += special_reward

        if action_str == "CloseObject" and state.metadata["lastObjectClosed"] is not None:
            special_reward = - self.target_receptacles_need_open.get(state.metadata["lastObjectOpened"]['objectType'], 0)
            # print('Closed {}, {} reward collected!'.format(opened_object['objectType'], special_reward))
            reward += special_reward

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            # print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        self.prev_inventory = []
        self.step_num = 0


class ExploreAllObjects(BaseTask):
    """
    This task consists of finding all objects in the enviorment.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.target_objects = kwargs['task'].get('target_objects', {})
        self.discoverd = set()

    def transition_reward(self, state, action_str=None):
        reward, done = self.movement_reward, False
        for obj in state.metadata['objects']:
            assert obj['name'] in self.target_objects
            if obj['visible'] and obj['name'] not in self.discoverd:
                self.discoverd.add(obj['name'])
                print("Found {} at {}, {}, {}".format(obj['name'], obj['position']['x'], obj['position']['y'], obj['position']['z']))
                reward += self.target_objects.get(obj['name'], 0)

        if self.max_episode_length and self.step_num >= self.max_episode_length or len(self.discoverd) == len(self.target_objects):
            if self.max_episode_length and self.step_num >= self.max_episode_length:
                print('Reached maximum episode length: {}'.format(self.step_num))
            else:
                print("Used {} steps to find all objects".format(self.step_num))
                reward += 50
            print('Totally found objects {}/{}'.format(len(self.discoverd), len(self.target_objects)))
            done = True

        return reward, done

    def reset(self):
        self.discoverd = set()
        self.step_num = 0
