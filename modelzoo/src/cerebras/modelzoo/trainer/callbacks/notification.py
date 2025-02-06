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
Contains callback that handles user notification on selected events.
"""

import os
import socket
from typing import List, Optional, Union

import requests

from cerebras.modelzoo.trainer.callbacks import Callback


class EmailNotification(Callback):
    """Callback for sending email notifications on certain events.

    Currently, the notification system requires an external notification server
    to be up and running which actually sends the emails. Please contact
    support@cerebras.net for setting up this server.
    """

    def __init__(
        self,
        mailto: Union[str, List[str]],
        notification_endpoint: Optional[str] = None,
    ):
        """
        Constructs an `EmailNotification` callback.

        Args:
            mailto: email address(es) to send notifications to.
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

        if notification_endpoint is None:
            notification_endpoint = os.environ.get(
                'CEREBRAS_NOTIFICATION_ENDPOINT'
            )
        if notification_endpoint is None:
            raise ValueError(
                "notification_endpoint argument or CEREBRAS_NOTIFICATION_ENDPOINT "
                "environment variable must be provided."
            )
        self.notification_endpoint = notification_endpoint

    def on_train_exception(self, trainer, exception):
        self.send_email_notification(
            trainer,
            f"Training failed due to error: {exception}. "
            f"Check {os.path.abspath(trainer.artifact_dir)} to see the "
            f"latest status of the job.",
        )

    def on_fit_end(self, trainer, loop):
        self.send_email_notification(
            trainer,
            f"Trainer fit completed successfully. Model directory is: "
            f"{os.path.abspath(trainer.model_dir)}",
        )

    def on_validate_exception(self, trainer, exception):
        self.send_email_notification(
            trainer,
            f"Validation failed due to error: {exception}. "
            f"Check {os.path.abspath(trainer.artifact_dir)} to see the "
            f"latest status of the job.",
        )

    def send_email_notification(self, trainer, message):
        notification_package = {
            'mailto': self.mailto,
            'subject': 'Cerebras Cluster Notification',
            'body': self.get_formatted_message(message),
        }
        try:
            response = requests.post(
                self.notification_endpoint,
                json=notification_package,
                timeout=5,
            )
            if response.status_code == 200:
                trainer.logger.verbose(
                    "Email notification was successfully sent."
                )
            else:
                trainer.logger.warning(
                    f"Failed to send email notification."
                    f"\nError code: {response.status_code}"
                    f"\nError content: {response.content}"
                )
        except Exception as e:
            trainer.logger.warning(
                f"Failed to send email notification due to error: {e}"
            )

    def get_formatted_message(self, message):
        return (
            f"Unix UserID: {os.environ.get('USER', 'NO_USER_ID')}\n"
            f"Hostname: {socket.gethostname()}\n"
            f"SystemMessage: {message}"
        )
