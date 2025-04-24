# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains callbacks that handle user notification on selected events.
"""

import abc
import os
import smtplib
import socket
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import List, Optional, Union

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from cerebras.modelzoo.trainer.callbacks import Callback


class Notification(Callback, abc.ABC):
    """Abstract base class for notification callbacks.

    Subclasses must implement `send_message`.
    """

    @abc.abstractmethod
    def send_message(self, message, logger):
        raise NotImplementedError()

    def _send_message(self, message, logger):
        formatted_message = (
            f"Unix UserID: {os.environ.get('USER', 'NO_USER_ID')}\n"
            f"Hostname: {socket.gethostname()}\n"
            f"SystemMessage: {message}"
        )
        self.send_message(formatted_message, logger)

    def on_fit_end(self, trainer, loop):
        self._send_message(
            f"Trainer fit completed successfully. Model directory is: "
            f"{os.path.realpath(trainer.model_dir)}",
            trainer.logger,
        )

    def on_train_exception(self, trainer, exception):
        self._send_message(
            f"Training failed due to error: {exception}. "
            f"Check {os.path.realpath(trainer.artifact_dir)} to see the latest status of the job.",
            trainer.logger,
        )

    def on_validate_exception(self, trainer, exception):
        self._send_message(
            f"Validation failed due to error: {exception}. "
            f"Check {os.path.realpath(trainer.artifact_dir)} to see the latest status of the job.",
            trainer.logger,
        )

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class EmailNotification(Notification):
    """Callback for sending email notifications on certain events.

    Currently, the notification system requires an external notification server
    to be up and running which actually sends the emails. Please contact
    support@cerebras.net for setting up this server.
    """

    def __init__(
        self,
        mailto: Union[str, List[str]],
        mailfrom: Optional[str] = None,
        server_name: Optional[str] = "smtp.office365.com",
        port: Optional[int] = 587,
        notification_endpoint: Optional[str] = None,
        subject: Optional[str] = "Cerebras Cluster Notification",
    ):
        """
        Constructs an `EmailNotification` callback.

        Requires setting the `CEREBRAS_NOTIFICATION_PASSWORD` environment variable to the
        password for the `mailfrom` email address.

        Args:
            mailto: email address(es) to send notifications to.
            mailfrom: email address to send notifications from. If not provided, its
                value is read from `CEREBRAS_NOTIFICATION_MAILFROM` environment variable.
                If still not provided, revert to using notification endpoint flow.
            notification_endpoint: A notification server that listens for requests which
                it then forwards to the recipients. If provided, this endpoint is used.
                Otherwise, its value is read from `CEREBRAS_NOTIFICATION_ENDPOINT`
                environment variable.
        """

        if isinstance(mailto, str):
            mailto = [mailto]
        if not isinstance(mailto, (list, tuple)) or any(
            not isinstance(m, str) for m in mailto
        ):
            raise ValueError(
                "mailto must be single recipient or a list of recipients."
            )
        if not mailto:
            raise ValueError("At least one recipient must be provided.")

        self.mailto = mailto
        self.subject = subject

        self.server = None

        if mailfrom is None:
            mailfrom = os.environ.get("CEREBRAS_NOTIFICATION_MAILFROM")
        self.mailfrom = mailfrom

        self.password = os.environ.get("CEREBRAS_NOTIFICATION_PASSWORD")

        if notification_endpoint is None:
            notification_endpoint = os.environ.get(
                'CEREBRAS_NOTIFICATION_ENDPOINT'
            )
        self.notification_endpoint = notification_endpoint

        if self.mailfrom:
            if self.notification_endpoint:
                raise ValueError(
                    f"Got values for both mailfrom ({self.mailfrom}) and "
                    f"notification_endpoint ({self.notification_endpoint}). "
                    f"You may only specify one."
                )

            if self.password is None:
                raise ValueError(
                    f"mailfrom was set but password was not. Please set the "
                    f"`CEREBRAS_NOTIFICATION_PASSWORD` environment variable."
                )

            # start server
            self.server = smtplib.SMTP(host=server_name, port=port)
            self.server.connect(server_name, port)
            self.server.ehlo()
            self.server.starttls()
            self.server.login(self.mailfrom, self.password)
        else:
            if self.notification_endpoint is None:
                raise ValueError(
                    "Both notification_endpoint and mailfrom are unset. One "
                    "of these two arguments/environment variables must be set."
                )

    def __del__(self):
        if self.server:
            self.server.quit()

    def send_message(self, message, logger):
        try:
            if self.server:
                msg = MIMEText(message)
                msg["Subject"] = self.subject
                msg["From"] = self.mailfrom
                msg["To"] = ",".join(self.mailto)

                self.server.sendmail(
                    self.mailfrom, self.mailto, msg.as_string()
                )
            else:
                notification_package = {
                    'mailto': self.mailto,
                    'subject': self.subject,
                    'body': message,
                }
                response = requests.post(
                    self.notification_endpoint,
                    json=notification_package,
                    timeout=5,
                )
                if response.status_code == 200:
                    logger.verbose("Email notification was successfully sent.")
                else:
                    logger.warning(
                        f"Failed to send email notification."
                        f"\nError code: {response.status_code}"
                        f"\nError content: {response.content}"
                    )
        except Exception as e:
            logger.warning(
                f"Failed to send email notification due to error: {e}"
            )


class SlackNotification(Notification):
    """Callback for sending slack notifications."""

    def __init__(self, channels: Union[str, List[str]]):
        """
        Constructs a `SlackNotification` callback.

        Requires setting the `CEREBRAS_SLACK_BOT_TOKEN` environment variable to the token
        of a slack bot that has been added to the provided slack channel.

        Args:
            channels: Slack channel(s) to send notifications to. Channels can be specified using their name or ID.
                To send notifications to a specific user, you can include their user ID in the list of channels.
                To find a user's ID, click the three dots in their Profile and select "Copy member ID".
        """
        if isinstance(channels, str):
            channels = [channels]
        if not isinstance(channels, (list, tuple)) or any(
            not isinstance(c, str) for c in channels
        ):
            raise ValueError(
                "`channels` must be a single channel/user or a list of channels/users."
            )
        self.channels = channels

        token = os.environ.get("CEREBRAS_SLACK_BOT_TOKEN")
        if not token:
            raise ValueError(
                "Bot token is required for the SlackNotification callback. "
                "Please set the `CEREBRAS_SLACK_BOT_TOKEN` env variable."
            )
        self.client = WebClient(token=token)

        for channel in self.channels:
            try:
                # Schedule a test message far in the future to make sure things are configured
                # correctly before we start a run and potentially fail later.
                response = self.client.chat_scheduleMessage(
                    channel=channel,
                    text="slack notification callback test message",
                    post_at=int(datetime.now(timezone.utc).timestamp())
                    + 600,  # 10 minutes in the future
                )
                self.client.chat_deleteScheduledMessage(
                    channel=channel,
                    scheduled_message_id=response["scheduled_message_id"],
                )
            except SlackApiError as e:
                if e.response["error"] == "not_in_channel":
                    raise RuntimeError(
                        "Failed to send slack notification. Bot has not been added to the channel "
                        f"'{channel}'. To add your bot to a channel, you can either mention the bot "
                        "in a message sent to the channel, or click the channel name in the channel "
                        "header and select the Integrations tab."
                    ) from e
                raise e

    def send_message(self, message, logger):
        for channel in self.channels:
            try:
                # Use scheduleMessage instead of postMessage because it supports sending
                # messages to channels using channel name, not just channel ID.
                self.client.chat_scheduleMessage(
                    channel=channel,
                    text=message,
                    post_at=int(datetime.now(timezone.utc).timestamp())
                    + 30,  # Add 30 seconds to avoid time_in_past errors
                )
            except Exception as e:
                logger.warning(
                    f"Failed sending slack notification to channel '{channel}' - {e}"
                )
