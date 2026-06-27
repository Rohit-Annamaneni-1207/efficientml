from abc import ABC, abstractmethod

class BaseCompressionModule(ABC):
    """
    Base class for compression modules.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.is_prepared = False
        self.is_applied = False

    @abstractmethod
    def prepare(self):
        """
        Prepare the compression module.
        """
        pass

    @abstractmethod
    def apply(self):
        """
        Apply the compression module.
        """
        pass

    @abstractmethod
    def export(self):
        """
        Export the compression module for deployment.

        Returns:
            The exported module.
        """
        pass

    @abstractmethod
    def summary(self):
        """
        Prepares a summary of the compression module.

        Returns:
            structured metadata
        """
        pass