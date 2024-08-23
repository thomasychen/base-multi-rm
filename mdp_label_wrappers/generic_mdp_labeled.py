from abc import ABC, abstractmethod

class MDP_Labeler(ABC):
    # @abstractmethod
    # def environment_step(self, s, a):
    #     """
    #      Execute action a from state s.

    #     Parameters
    #     ----------
    #     s : int
    #         Index representing the current environment state
    #     a : int
    #         Index representing the action being taken

    #     Return
    #     ---------
    #     s_next : int
    #         Index of next state
    #     """
    #     raise NotImplementedError
    
    @abstractmethod
    def get_mdp_label(self, s_next):
        """
        Return the label of the next environment state and current RM state.
        """
        raise NotImplementedError