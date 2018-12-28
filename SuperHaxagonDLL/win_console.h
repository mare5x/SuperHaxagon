#pragma once

void open_console();

void close_console();

void hide_console();

void show_console();

/* Refer to [https://docs.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences#text-formatting] for possible <value> options. */
void set_text_formatting(int value);