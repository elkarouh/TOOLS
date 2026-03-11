"""
hblackboard - A pub/sub blackboard pattern implementation.

Provides a PublishChannel class that enables decoupled communication
between knowledge sources via named event channels. Subscribers register
callbacks on channels and are notified when events are published.
"""
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
# =============================================================================
# PUB/SUB INFRASTRUCTURE
# =============================================================================
class PublishChannel:
    """
    Enhanced pub/sub event channel for blackboard communication.
    Supports both sync and async subscribers.
    """
    def __init__(self, name: str):
        """
        Initialize a publish channel.

        Args:
            name: Identifier for this channel, passed to subscribers on publish.
        """
        self.name = name
        self.subscribers: List[Callable] = []
        self.history: List[Dict] = []
        self.keep_history: bool = False
        self.max_history: int = 1000

    def subscribe(self, callback: Callable) -> None:
        """
        Subscribe a callback to this channel.

        Args:
            callback: A callable invoked with (channel_name, **kwargs) on publish.
                      Duplicate subscriptions are ignored.
        """
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            logger.debug(f"Subscriber {callback.__name__} added to {self.name}")

    def unsubscribe(self, callback: Callable) -> None:
        """
        Unsubscribe a callback from this channel.

        Args:
            callback: The previously subscribed callable to remove.
                      No-op if the callback is not currently subscribed.
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug(f"Subscriber {callback.__name__} removed from {self.name}")

    def publish(self, **kwargs) -> int:
        """
        Publish an event to all subscribers.

        Each subscriber is called with the channel name as the first positional
        argument, followed by all keyword arguments passed here. Errors in
        individual subscribers are logged and do not prevent other subscribers
        from being notified.

        Args:
            **kwargs: Arbitrary keyword arguments forwarded to each subscriber.

        Returns:
            The number of subscribers successfully notified.
        """
        if self.keep_history:
            self.history.append({
                'timestamp': datetime.now(),
                'data': kwargs
            })
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

        notified = 0
        for callback in self.subscribers:
            try:
                callback(self.name, **kwargs)
                notified += 1
            except Exception as e:
                logger.error(f"Error in subscriber {callback.__name__} on {self.name}: {e}")

        return notified

    def clear_subscribers(self) -> None:
        """Remove all subscribers from this channel."""
        self.subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        """Return the number of currently registered subscribers."""
        return len(self.subscribers)


#############################################################
if __name__=="__main__":
    class Blackboard:
        """Central shared data structure that holds the publish channels."""
        def __init__(self,name="BLACKBOARD"):
            """Initialize the blackboard with three alarm channels."""
            self.name=name
            self.alarm_channel_1 = PublishChannel ("alarm_channel_1")
            self.alarm_channel_2 = PublishChannel ("alarm_channel_2")
            self.alarm_channel_3 = PublishChannel ("alarm_channel_3")
    class KnowledgeSource_1:
        """Knowledge source subscribed to alarm channels 1 and 2."""
        def __init__ (self, blackboard):
            self.blackboard=blackboard
            blackboard.alarm_channel_1.subscribe (self.accept_alarm1)
            blackboard.alarm_channel_2.subscribe (self.accept_alarm2)
        def accept_alarm1 (self, channel,*, first="None", second="None"):
            """The first argument must be the channel blackboard
            Many arguments may follow.
            """
            print (self.blackboard.name,"received signal:", first, second,"on channel",channel)
        def accept_alarm2 (self, channel,*, first="None", second="None"):
            """The first argument must be the channel blackboard
            Many arguments may follow.
            """
            print (self.blackboard.name,"received signal:", first, second,"on channel",channel)
    class KnowledgeSource_2:
        """Knowledge source subscribed to alarm channels 2 and 3."""
        def __init__(self, blackboard):
            self.blackboard=blackboard
            blackboard.alarm_channel_2.subscribe (self.accept_alarm2)
            blackboard.alarm_channel_3.subscribe (self.accept_alarm3)
        def accept_alarm2 (self, channel,*, first="None", second="None"):
            """The first argument must be the channel blackboard
            Many arguments may follow.
            """
            print (self.blackboard.name,"received signal:", first, second,"on channel",channel)
        def accept_alarm3 (self, channel,*, first="None"):
            """The first argument must be the channel blackboard
            Many arguments may follow.
            """
            print (self.blackboard.name,"received signal:", first, "on channel",channel)

    b=Blackboard()
    KnowledgeSource_1(b)
    KnowledgeSource_2(b)
    ######################### USAGE ###################
    b.alarm_channel_1.publish(first="message for chan1",second=111)
    b.alarm_channel_2.publish(first="message for chan2",second=222)
    b.alarm_channel_3.publish(first="message for chan3")
