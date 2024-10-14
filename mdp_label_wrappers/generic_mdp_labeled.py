from abc import ABC, abstractmethod

class MDP_Labeler(ABC):
    
    @abstractmethod
    def get_mdp_label(self, s_next):
        """
        Return the label of the next environment state and current RM state.
        """
        raise NotImplementedError