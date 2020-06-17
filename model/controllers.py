class Controller(object):
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_action(self, state, evaluation=False):
        return self.env.action_space.sample()


class PolicyController(Controller):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def get_action(self, state, evaluation=False):
        action, mean_logstd = self.policy.get_action(state)
        if evaluation:
            return mean_logstd["mean"][0]
        else:
            return action

    def get_actions(self, states):
        actions, _ = self.policy.get_actions(states)
        return actions

    def get_actions_tf(self, states):
        actions, _ = self.policy.get_actions_tf(states)
        return actions
