from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

class EmailData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: int | None = Field(default=None, alias='id')
    message_id: str | None = Field(default=None, alias='message_id')
    date: datetime | None = Field(default=None, alias='date')
    sender: str | None = Field(default=None, alias='from')
    to: list[str] | None = Field(default=None, alias='to')
    cc: list[str] | None = Field(default=None, alias='cc')
    bcc: list[str] | None = Field(default=None, alias='bcc')
    subject: str | None = Field(default=None, alias='subject')
    body: str | None = Field(default=None, alias='body')
    x_from: str | None = Field(default=None, alias='x_from')
    x_to: str | None = Field(default=None, alias='x_to')
    x_cc: str | None = Field(default=None, alias='x_cc')
    x_bcc: str | None = Field(default=None, alias='x_bcc')
    x_folder: str | None = Field(default=None, alias='x_folder')
    x_origin: str | None = Field(default=None, alias='x_origin')
    x_filename: str | None = Field(default=None, alias='x_filename')
    source_file: str | None = Field(default=None, alias='source_file')

    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, date: object) -> datetime | None:
        return parsedate_to_datetime(date).astimezone(timezone.utc) if isinstance(date, str) else date

    @field_validator('to', 'cc', 'bcc', mode='before')
    @classmethod
    def parse_email(cls, value: object) -> list[str] | None:
        email_patterns = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        return sorted({email.lower() for email in email_patterns.findall(value)}) or None if isinstance(value, str) else value

class ThreadData(EmailData):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    thread_id: str
    parent_id: str | None = None
    sequence: int
    is_forward: bool
    is_reply: bool

    sender_domain: str | None = None
    recipient_domain: list[str] | None = None

